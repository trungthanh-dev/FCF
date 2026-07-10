import torch
import torch.nn as nn
import os
import sys
from torch.nn.functional import dropout
from torch.utils.checkpoint import checkpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import TensorDataset, DataLoader

from config import RANDOM_STATE

class _LSTMNet(nn.Module):
    def __int__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers >1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size,1)

    def forward(self, x):
        out, _=self.lstm(x)
        last_hidden = out[:,-1,:]
        return self.fc(last_hidden).squeeze(-1)

class LSTMModel:
    def __init__(
            self,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=1e-3,
            epochs=20,
            batch_size=64,
            device=None,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(RANDOM_STATE)

        self.model = None

    def _build_model(self, input_size):
        self.model = _LSTMNet(
            input_size=input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = self.dropout,
        ).tp(self.device)

    def train(
            self,
            X_train,
            y_train,
            verbose = True,
    ):
        if self.model is None:
            self._build_model(input_size=X_train.shape[2])

        X_tensor = torch.tensor(X_train, dtype = torch.float32)
        y_tensor = torch.tensor(y_train, dtype = torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()*X_batch.size(0)

            epoch_loss = total_loss/len(dataset)
            if verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss:.6f}")

    def predict(
            self,
            X_test,
    ):
        self.model.eval()
        X_tensor = torch.tensor(X_test, dtypr=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = self.model(X_tensor)

        return y_pred.cpu().numpy()

    def save(
            self,
            path,
    ):
        torch.save(
            {
                "State_dict": self.model.state_dict(),
                "hidden_state": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "input_size": self.model.input_size,
            },
            path,
        )

    def load(
            self,
            path,
    ):
        checkpoint= torch.load(path, map_location= self.device)

        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]

        self._build_model(input_size=checkpoint["input_size"])
        self.model.load_state_dict(checkpoint["state_dict"])
