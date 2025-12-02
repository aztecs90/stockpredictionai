import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from ai_trader.config.settings import MODELS_DIR

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, num_classes=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid() # For binary classification (Up/Down) or use Linear for regression

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class ModelEngine:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_sequences(self, data, seq_length, target_col='Returns'):
        """
        Creates sequences for LSTM.
        Target: 1 if next candle Returns > 0 else 0 (Binary Classification for Swing High)
        """
        # Feature columns are all except target-like stuff if we were doing regression on price
        # For this logic, we use all technicals as input
        feature_data = data.values

        X, y = [], []
        # Create Target: Next day's return positive?
        # Shift -1 to compare current close with next close
        # If Close[t+1] > Close[t] -> Buy (1)
        targets = (data['Close'].shift(-1) > data['Close']).astype(int).values

        for i in range(len(feature_data) - seq_length - 1):
            X.append(feature_data[i:(i + seq_length)])
            y.append(targets[i + seq_length])

        return np.array(X), np.array(y)

    def train(self, df, model_name, config, callback=None):
        """
        df: Processed dataframe
        config: dict with epochs, lr, hidden_size, etc.
        callback: function to report progress (epoch, loss)
        """
        seq_length = config.get('sequence_length', 10)
        epochs = config.get('epochs', 50)
        lr = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 32)
        hidden_size = config.get('hidden_size', 64)
        num_layers = config.get('num_layers', 1)

        # Scale Data
        scaled_data = self.scaler.fit_transform(df)

        # Prepare Sequences
        X, y = self.prepare_sequences(pd.DataFrame(scaled_data, columns=df.columns), seq_length)

        # Split Train/Test
        split_idx = int(len(X) * config.get('train_split', 0.8))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # To Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Init Model
        input_size = X.shape[2]
        self.model = LSTMModel(input_size, hidden_size, num_layers).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training Loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if callback:
                callback(epoch + 1, avg_loss)
            else:
                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save Model & Scaler
        self.save_model(model_name)

        return self.evaluate(X_test_t, y_test_t)

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
        return accuracy.item()

    def predict(self, df, config):
        """Generates predictions for the whole dataframe to be used in backtesting."""
        if not self.model:
            raise ValueError("Model not trained or loaded")

        seq_length = config.get('sequence_length', 10)
        scaled_data = self.scaler.transform(df)

        # We need to predict for every possible day.
        # But sequences consume 'seq_length' initial days.
        # We will pad the beginning with NaNs or False to match index length

        X_all = []
        for i in range(len(scaled_data) - seq_length):
            X_all.append(scaled_data[i:i+seq_length])

        X_tensor = torch.FloatTensor(np.array(X_all)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            probs = outputs.cpu().numpy().flatten()

        # Pad initial rows
        padding = [0] * seq_length
        full_preds = np.concatenate((padding, preds))
        full_probs = np.concatenate((padding, probs))

        # Ensure length matches df
        # Note: prepare_sequences logic in train used shift(-1), so predictions are for "Next Day"
        # If index i has prediction 1, it means at the END of day i, we predict day i+1 is UP.
        # So we signal a BUY at Close of day i.

        if len(full_preds) < len(df):
             # This might happen due to the way loops range
             full_preds = np.pad(full_preds, (0, len(df) - len(full_preds)))
             full_probs = np.pad(full_probs, (0, len(df) - len(full_probs)))

        return full_preds, full_probs

    def save_model(self, name):
        torch.save(self.model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pth"))
        with open(os.path.join(MODELS_DIR, f"{name}_scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self, name, input_size, config):
        path = os.path.join(MODELS_DIR, f"{name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {name} not found")

        hidden_size = config.get('hidden_size', 64)
        num_layers = config.get('num_layers', 1)

        self.model = LSTMModel(input_size, hidden_size, num_layers).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        with open(os.path.join(MODELS_DIR, f"{name}_scaler.pkl"), 'rb') as f:
            self.scaler = pickle.load(f)
