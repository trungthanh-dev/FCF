import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE


class _Seq2SeqNet(nn.Module):
    """
    Encoder-decoder LSTM.

    Encoder: reads the WINDOW_SIZE past steps, produces a final
    (hidden, cell) state that summarizes the whole window.

    Decoder: an autoregressive LSTM that starts from the encoder's final
    state and unrolls for n_horizons steps. At each step it consumes its
    OWN previous prediction (starting from a learned "start" value), and
    outputs the next horizon's prediction. Because every horizon comes
    from the same decoder state trajectory, predictions across horizons
    share one consistent internal representation -- unlike training 4
    separate direct-forecast models, whose tree/network splits never
    have to agree with each other.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, n_horizons):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_horizons = n_horizons

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Decoder input at each step: the previous step's predicted value
        # (scalar). Autoregressive, like a standard seq2seq decoder.
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # Learned "start token" for the first decoder input, instead of
        # an arbitrary constant like 0.
        self.start_token = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, x):
        batch_size = x.size(0)
        _, (h, c) = self.encoder(x)

        decoder_input = self.start_token.expand(batch_size, 1, 1)
        hidden = (h, c)

        outputs = []
        for _ in range(self.n_horizons):
            out, hidden = self.decoder(decoder_input, hidden)
            out = self.dropout_layer(out.squeeze(1))
            pred = self.output_layer(out)          # (batch, 1)
            outputs.append(pred)
            decoder_input = pred.unsqueeze(1)       # feed prediction back in

        return torch.cat(outputs, dim=1)            # (batch, n_horizons)


class Seq2SeqLSTMModel:
    """
    Same conventions as models/lstm.py's LSTMModel: X scaling fit on
    train only, a separate target scaler (fit across ALL horizon
    columns jointly, so relative magnitude between horizons is
    preserved), Huber loss, gradient clipping, chronological val split,
    early stopping, per-instance val_ratio/patience so callers (e.g.
    experiments.py) can set per-ship hyperparameters exactly like the
    direct-forecast LSTM.
    """

    def __init__(
            self,
            horizons,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            learning_rate=5e-4,
            epochs=150,
            batch_size=128,
            val_ratio=0.1,
            patience=10,
            loss_delta=1.0,
            device=None,
    ):
        self.horizons = list(horizons)
        self.n_horizons = len(self.horizons)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.patience = patience
        self.loss_delta = loss_delta

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_STATE)
            torch.cuda.manual_seed_all(RANDOM_STATE)

        self.model = None
        self.scaler = StandardScaler()
        # One scaler for the target, fit jointly across all horizon
        # columns (not one scaler per horizon) so the decoder learns a
        # single consistent output scale across the whole sequence.
        self.y_scaler = StandardScaler()

    def _scale_fit(self, X):
        n_samples, w, f = X.shape
        X_2d = X.reshape(-1, f)
        self.scaler.fit(X_2d)
        return self.scaler.transform(X_2d).reshape(n_samples, w, f)

    def _scale_transform(self, X):
        n_samples, w, f = X.shape
        X_2d = X.reshape(-1, f)
        return self.scaler.transform(X_2d).reshape(n_samples, w, f)

    def _y_scale_fit(self, y):
        # y: (samples, n_horizons). Fit on all horizon columns jointly by
        # flattening, so a single scale/offset applies to every horizon.
        flat = y.reshape(-1, 1)
        self.y_scaler.fit(flat)
        return self.y_scaler.transform(flat).reshape(y.shape)

    def _y_scale_transform(self, y):
        flat = y.reshape(-1, 1)
        return self.y_scaler.transform(flat).reshape(y.shape)

    def _y_inverse_transform(self, y_scaled):
        shape = y_scaled.shape
        flat = y_scaled.reshape(-1, 1)
        return self.y_scaler.inverse_transform(flat).reshape(shape)

    def _build_model(self, input_size):
        self.model = _Seq2SeqNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_horizons=self.n_horizons,
        ).to(self.device)

    def train(self, X_train, y_train, verbose=True, val_ratio=None, patience=None):
        if val_ratio is None:
            val_ratio = self.val_ratio
        if patience is None:
            patience = self.patience

        if self.model is None:
            self._build_model(input_size=X_train.shape[2])

        n_val = int(len(X_train) * val_ratio)
        X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

        X_tr_scaled = self._scale_fit(X_tr)
        X_val_scaled = self._scale_transform(X_val)

        y_tr_scaled = self._y_scale_fit(y_tr)
        y_val_scaled = self._y_scale_transform(y_val)

        X_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_tr_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.HuberLoss(delta=self.loss_delta)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)          # (batch, n_horizons)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

            train_loss = total_loss / len(dataset)

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor).item()
            scheduler.step(val_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1} "
                              f"(no val improvement for {patience} epochs)")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_test):
        """
        Returns predictions with shape (samples, n_horizons), in the same
        order as self.horizons -- e.g. column 0 = horizon 1, column 1 =
        horizon 5, etc. (whatever order was passed to __init__).
        """
        self.model.eval()
        X_test_scaled = self._scale_transform(X_test)
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor)

        y_pred_scaled = y_pred_scaled.cpu().numpy()
        return self._y_inverse_transform(y_pred_scaled)

    def save(self, path):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "input_size": self.model.input_size,
                "horizons": self.horizons,
                "scaler": self.scaler,
                "y_scaler": self.y_scaler,
            },
            path,
        )

    def load(self, path):
        # weights_only=False needed because the checkpoint also stores
        # sklearn StandardScaler objects (not just tensors) -- PyTorch
        # 2.6+ defaults torch.load to weights_only=True, which would
        # otherwise reject them. Only load checkpoints you trust.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.horizons = checkpoint["horizons"]
        self.n_horizons = len(self.horizons)
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.scaler = checkpoint["scaler"]
        self.y_scaler = checkpoint["y_scaler"]
        self._build_model(input_size=checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["state_dict"])