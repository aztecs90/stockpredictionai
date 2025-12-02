import os
import json
import sys

# Define the base directory (root of the project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "ai_trader" not in BASE_DIR: # Fallback if structure is different
    BASE_DIR = os.getcwd()

# Data and Artifacts Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEFAULT_SETTINGS = {
    "tickers": ["RELIANCE.NS", "^NSEI", "MON100.NS"],
    "date_range": {"start": "2020-01-01", "end": None}, # None means today
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_size": 64,
        "num_layers": 1,
        "sequence_length": 10,
        "train_split": 0.8
    },
    "features": {
        "enabled": [
            "Returns", "RSI", "MACD", "Signal", "Hist",
            "BB_Upper", "BB_Lower", "BB_Mid", "ADX", "CCI", "ATR"
        ],
        "use_vix": True
    },
    "backtest": {
        "initial_capital": 100000,
        "fees": 0.001,
        "slippage": 0.001
    }
}

class SettingsManager:
    def __init__(self):
        self.settings = DEFAULT_SETTINGS.copy()
        self.load_settings()

    def load_settings(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    user_settings = json.load(f)
                    # Deep update logic could be better, but simple update for now
                    self.settings.update(user_settings)
            except Exception as e:
                print(f"Failed to load settings: {e}")

    def save_settings(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = value
        self.save_settings()

# Global instance
settings_manager = SettingsManager()
