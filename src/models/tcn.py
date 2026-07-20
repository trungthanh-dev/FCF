import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_STATE
from evalute import dtw_distance


class _Chomp1d(nn.Module):
    """
    Causal convolution with padding on both sides (needed so the output
    length matches the input length) ends up with a few extra timesteps
    on the RIGHT that "see into the future" relative to what we want.
    This layer just chops those off, so each output timestep only ever
    depends on itself and earlier timesteps -- the causal property.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    """
    One residual block: two causal dilated conv1d layers, each followed
    by chomp (restore causality) -> ReLU -> dropout, plus a residual
    ("skip") connection from the block's input to its output. If the
    number of channels changes between input and output, a 1x1 conv
    projects the input so the residual add still works.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding: pad left only, in effect

        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        # 1x1 conv to match channel dimensions for the residual add, only
        # created when needed (n_inputs != n_outputs).
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Bai et al. (2018) initialize conv weights from N(0, 0.01), not
        # PyTorch's default fan_in-based init. Without this, weight_norm's
        # initial magnitude (computed from the default-init direction
        # vector) is roughly an order of magnitude too large, which makes
        # training chaotically sensitive to CPU floating-point rounding
        # noise from multi-threaded conv ops -- observed empirically as R2
        # swinging between ~0.04 and ~0.29 for the same code/config across
        # separate process runs. The small-std init keeps initial
        # activations/gradients well-scaled and removes that instability.
        #
        # conv1/conv2 go through the `weight_norm` parametrization, so
        # `.weight` is a *computed* tensor (g * v/||v||), not the stored
        # parameter -- writing to `.weight.data` is silently a no-op. The
        # actual learnable tensors live at `.parametrizations.weight.
        # original0` (g, per-output-channel magnitude) and `.original1`
        # (v, direction). Set v ~ N(0, 0.01) then set g = ||v|| so the
        # computed weight g*v/||v|| equals v exactly at init time.
        with torch.no_grad():
            for conv in (self.conv1, self.conv2):
                v = conv.parametrizations.weight.original1
                v.normal_(0, 0.01)
                g = conv.parametrizations.weight.original0
                g.copy_(v.norm(dim=(1, 2), keepdim=True))
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TCNNet(nn.Module):
    """
    Stack of _TemporalBlock layers with exponentially increasing dilation
    (1, 2, 4, 8, ...), so the receptive field grows fast without needing
    a very deep or wide network. After the stack, only the LAST timestep
    (the one closest to "now") is used to produce a single scalar
    prediction -- analogous to how an LSTM's final hidden state is used.
    """
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.input_size = input_size
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x arrives as (batch, window_size, features) -- Conv1d wants
        # (batch, channels, length), so features must come before time.
        x = x.transpose(1, 2)          # (batch, features, window_size)
        out = self.network(x)          # (batch, channels, window_size)
        last_step = out[:, :, -1]      # (batch, channels) -- most recent timestep
        return self.output_layer(last_step).squeeze(-1)


class TCNModel:
    """
    Same conventions as models/lstm.py's LSTMModel: X/y scaling fit on
    train only, Huber loss, gradient clipping, chronological val split,
    early stopping, per-instance val_ratio/patience for per-ship tuning.
    One model = one horizon (direct forecasting), same as LSTMModel, so
    results compare directly against the existing LSTM/RF/XGBoost tables.
    """

    def __init__(
            self,
            num_channels=(32, 32, 32, 32),
            kernel_size=3,
            dropout=0.1,
            learning_rate=5e-4,
            epochs=150,
            batch_size=128,
            val_ratio=0.1,
            patience=10,
            loss_delta=1.0,
            weight_decay=1e-5,
            adam_eps=1e-4,
            early_stop_metric="huber",
            dtw_window=10,
            dtw_weight=0.5,
            device=None,
    ):
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.patience = patience
        self.loss_delta = loss_delta
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        assert early_stop_metric in ("huber", "dtw", "combined")
        self.early_stop_metric = early_stop_metric
        self.dtw_window = dtw_window
        self.dtw_weight = dtw_weight

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_STATE)
            torch.cuda.manual_seed_all(RANDOM_STATE)

        self.model = None
        self.scaler = StandardScaler()
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
        self.model = _TCNNet(
            input_size=input_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
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
        # eps default of 1e-4 (PyTorch's Adam default is 1e-8): with mostly
        # near-zero gradients -- e.g. delta targets that are ~0 for long
        # flat stretches -- Adam's per-parameter second-moment estimate can
        # shrink enough that sqrt(v_hat)+eps is dominated by eps itself; too
        # small an eps then blows the update up (and, combined with
        # weight_norm's own division by ||v||, to NaN) the moment a batch
        # with real signal appears. Verified empirically: eps=1e-8 corrupted
        # ~12% of batches with non-finite gradients on Poseidon delta-target
        # training; eps=1e-4 had zero.
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay, eps=self.adam_eps,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            n_seen = 0
            n_skipped = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # weight_norm's backward divides by ||v|| (the direction
                # vector's norm); if that norm collapses toward zero the
                # gradient can blow up to NaN/Inf for a single batch (seen
                # empirically with delta-target training, where most
                # batches sit near zero and a few extreme-outlier batches
                # spike the loss). Stepping on a NaN/Inf gradient would
                # permanently corrupt every weight for the rest of training
                # (irreversible, not something clipping can fix after the
                # fact) -- so skip just that batch's update instead.
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    n_skipped += 1
                    continue
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
                n_seen += X_batch.size(0)

            train_loss = total_loss / n_seen if n_seen else float("nan")
            if verbose and n_skipped:
                print(f"  ({n_skipped} batch(es) skipped this epoch: non-finite gradient)")

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor).item()
            scheduler.step(val_loss)

            if self.early_stop_metric in ("dtw", "combined"):
                # Val split is the last val_ratio slice of X_train, kept in
                # chronological order (no shuffling), so DTW over it is a
                # legitimate sequence comparison -- same as the eval-time
                # DTW computed on y_test/y_pred, just on the validation
                # window instead. Huber loss above still drives the LR
                # scheduler; only checkpoint selection swaps metric.
                val_pred_raw = self._y_inverse_transform(val_pred.cpu().numpy())
                y_val_raw = self._y_inverse_transform(y_val_tensor.cpu().numpy())
                val_dtw = dtw_distance(y_val_raw, val_pred_raw, window=self.dtw_window)

                if self.early_stop_metric == "dtw":
                    select_metric = val_dtw
                    metric_note = f"  val_dtw: {val_dtw:.6f}"
                else:
                    val_mae = float(np.mean(np.abs(y_val_raw - val_pred_raw)))
                    select_metric = (1 - self.dtw_weight) * val_mae + self.dtw_weight * val_dtw
                    metric_note = f"  val_mae: {val_mae:.6f}  val_dtw: {val_dtw:.6f}  val_combined: {select_metric:.6f}"
            else:
                select_metric = val_loss
                metric_note = ""

            if verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}{metric_note}")

            if select_metric < best_val_loss:
                best_val_loss = select_metric
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
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "input_size": self.model.input_size,
                "scaler": self.scaler,
                "y_scaler": self.y_scaler,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.num_channels = checkpoint["num_channels"]
        self.kernel_size = checkpoint["kernel_size"]
        self.dropout = checkpoint["dropout"]
        self.scaler = checkpoint["scaler"]
        self.y_scaler = checkpoint["y_scaler"]
        self._build_model(input_size=checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["state_dict"])