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


class _LSTMNet(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout):
        super().__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,

        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout_layer(last_hidden)
        return self.fc(last_hidden).squeeze(-1)


class LSTMModel:
    def __init__(
            self,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            learning_rate=5e-4,
            epochs=150,
            batch_size=128,
            val_ratio=0.1,
            patience=10,
            device=None,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # FIX: val_ratio/patience are now configurable per model instance
        # (previously hardcoded as train() defaults, so every ship was
        # forced to use the same early-stopping behavior regardless of
        # dataset size — Triton/Ceto are much smaller than Poseidon and
        # overfit much faster, so they need their own patience/val_ratio).
        self.val_ratio = val_ratio
        self.patience = patience

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_STATE)
            torch.cuda.manual_seed_all(RANDOM_STATE)
        self.model = None
        self.scaler = StandardScaler()
        # FIX: added a separate scaler for the target. Previously X was
        # scaled but y was left in raw units. For ships like Triton/Ceto,
        # where fuel values sit in a small range (~0.02-0.35), MSE loss on
        # unscaled y produces very small gradients, which combined with a
        # conservative learning_rate (5e-4) and early stopping (patience=10)
        # can cause training to stop before the model has learned anything
        # beyond predicting close to the mean — consistent with the
        # near-identical R2 seen at h=10 and h=20 for Triton.
        self.y_scaler = StandardScaler()

    def _scale_fit(self, X):
        # X: (samples, window, features) -> fit trên toàn bộ điểm dữ liệu train
        n_samples, w, f = X.shape
        X_2d = X.reshape(-1, f)
        self.scaler.fit(X_2d)
        return self.scaler.transform(X_2d).reshape(n_samples, w, f)

    def _scale_transform(self, X):
        n_samples, w, f = X.shape
        X_2d = X.reshape(-1, f)
        return self.scaler.transform(X_2d).reshape(n_samples, w, f)

    def _y_scale_fit(self, y):
        y_2d = np.asarray(y).reshape(-1, 1)
        self.y_scaler.fit(y_2d)
        return self.y_scaler.transform(y_2d).reshape(-1)

    def _y_scale_transform(self, y):
        y_2d = np.asarray(y).reshape(-1, 1)
        return self.y_scaler.transform(y_2d).reshape(-1)

    def _y_inverse_transform(self, y_scaled):
        y_2d = np.asarray(y_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(y_2d).reshape(-1)

    def _build_model(self, input_size):
        self.model = _LSTMNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def train(self, X_train, y_train, verbose=True, val_ratio=None, patience=None):
        # FIX: fall back to the instance's own val_ratio/patience (set in
        # __init__) when not explicitly passed here, so callers like
        # experiments.py can configure early stopping per LSTMModel
        # instance instead of relying on a single hardcoded default.
        if val_ratio is None:
            val_ratio = self.val_ratio
        if patience is None:
            patience = self.patience

        if self.model is None:
            self._build_model(input_size=X_train.shape[2])

        # Tách validation từ 10% CUỐI của train (giữ nguyên thứ tự thời gian,
        # không shuffle trước khi tách) -- để có tín hiệu dừng sớm, tránh
        # overfit như trường hợp Triton (train loss giảm đều nhưng test R2 âm).
        n_val = int(len(X_train) * val_ratio)
        X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

        X_tr_scaled = self._scale_fit(X_tr)      # fit scaler CHỈ trên phần train thật
        X_val_scaled = self._scale_transform(X_val)

        # FIX: fit the target scaler on y_tr only (same rule as X — never
        # fit on validation or test data), then scale y_tr/y_val to match
        # the scale the network trains in. Predictions are inverse-transformed
        # back to raw fuel units in predict().
        y_tr_scaled = self._y_scale_fit(y_tr)
        y_val_scaled = self._y_scale_transform(y_val)

        X_tensor = torch.tensor(X_tr_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_tr_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
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
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

            train_loss = total_loss / len(dataset)

            # --- validation loss, để biết có đang overfit không ---
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor).item()
            scheduler.step(val_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

            # --- early stopping ---
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
            self.model.load_state_dict(best_state)  # khôi phục weight tốt nhất, không phải weight cuối cùng

    def predict(self, X_test):
        self.model.eval()
        X_test_scaled = self._scale_transform(X_test)  # dùng scaler đã fit từ train
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor)

        # FIX: model now outputs predictions in scaled (standardized) units,
        # since it was trained on scaled y. Inverse-transform back to raw
        # fuel units before returning, so evaluate_regression() compares
        # like-for-like against the unscaled y_test used elsewhere in the
        # pipeline (e.g. Random Forest results).
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
                "scaler": self.scaler,
                "y_scaler": self.y_scaler,   # FIX: must be saved too, or load() can't inverse-transform predictions
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.scaler = checkpoint["scaler"]
        self.y_scaler = checkpoint["y_scaler"]   # FIX: restore target scaler alongside feature scaler
        self._build_model(input_size=checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["state_dict"])