from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QFormLayout,
                             QLineEdit, QCheckBox, QPushButton, QHBoxLayout,
                             QGroupBox, QScrollArea, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt
from ai_trader.config.settings import settings_manager
from ai_trader.utils.workers import PipelineWorker

class AdvancedView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Tab 1: Configuration & Features ---
        self.config_tab = QWidget()
        self.setup_config_tab()
        self.tabs.addTab(self.config_tab, "Configuration")

        # --- Tab 2: Model Training ---
        self.train_tab = QWidget()
        self.setup_train_tab()
        self.tabs.addTab(self.train_tab, "Training & Tuning")

        self.setLayout(layout)

    def setup_config_tab(self):
        layout = QVBoxLayout(self.config_tab)

        # General Settings
        form_layout = QFormLayout()
        self.ticker_input = QLineEdit("RELIANCE.NS, ^NSEI")
        form_layout.addRow("Tickers (comma sep):", self.ticker_input)
        layout.addLayout(form_layout)

        # Feature Selection
        feat_group = QGroupBox("Dynamic Feature Selection")
        feat_layout = QVBoxLayout()

        self.feature_checks = {}
        # List of potential indicators the DataProcessor supports
        available_features = [
            "RSI", "MACD", "Signal", "Hist", "EMA_20", "SMA_50", "SMA_200",
            "Dist_EMA200", "ADX", "Stoch_K", "Stoch_D", "CCI",
            "BB_Upper", "BB_Lower", "BB_Mid", "ATR", "MFI"
        ]

        # Grid layout for checkboxes
        grid = QHBoxLayout() # Actually let's use flow-like layout or grid
        # Simple implementation: Wrap in scroll area if many
        scroll = QScrollArea()
        scroll_widget = QWidget()
        grid_layout = QVBoxLayout(scroll_widget)

        current_enabled = settings_manager.get("features", {}).get("enabled", [])

        for feat in available_features:
            chk = QCheckBox(feat)
            if feat in current_enabled:
                chk.setChecked(True)
            self.feature_checks[feat] = chk
            grid_layout.addWidget(chk)

        self.vix_check = QCheckBox("Include VIX Index")
        self.vix_check.setChecked(settings_manager.get("features", {}).get("use_vix", True))
        grid_layout.addWidget(self.vix_check)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        feat_layout.addWidget(scroll)
        feat_group.setLayout(feat_layout)
        layout.addWidget(feat_group)

        # Save Button
        btn_save = QPushButton("Save Configuration")
        btn_save.clicked.connect(self.save_config)
        layout.addWidget(btn_save)

    def setup_train_tab(self):
        layout = QVBoxLayout(self.train_tab)

        form_layout = QFormLayout()

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(settings_manager.get("training", {}).get("epochs", 50))

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(4)
        self.lr_input.setRange(0.0001, 0.1)
        self.lr_input.setValue(settings_manager.get("training", {}).get("learning_rate", 0.001))

        form_layout.addRow("Epochs:", self.epochs_input)
        form_layout.addRow("Learning Rate:", self.lr_input)

        layout.addLayout(form_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_tune = QPushButton("Run Auto-Tuner")
        self.btn_train = QPushButton("Train Selected Ticker")

        btn_layout.addWidget(self.btn_tune)
        btn_layout.addWidget(self.btn_train)
        layout.addLayout(btn_layout)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        # Connect
        self.btn_tune.clicked.connect(self.run_tuner)
        self.btn_train.clicked.connect(self.run_train)

    def save_config(self):
        enabled = [k for k, v in self.feature_checks.items() if v.isChecked()]

        features_settings = {
            "enabled": enabled,
            "use_vix": self.vix_check.isChecked()
        }

        settings_manager.set("features", features_settings)
        # Also update training params from tab 2 if changed
        train_settings = settings_manager.get("training", {})
        train_settings["epochs"] = self.epochs_input.value()
        train_settings["learning_rate"] = self.lr_input.value()
        settings_manager.set("training", train_settings)

        QMessageBox.information(self, "Saved", "Configuration saved successfully.")

    def run_tuner(self):
        # Implementation to invoke worker with tuning mode
        self.log_area.append("Starting Hyperparameter Tuner...")
        # (Simplified: In a real app we'd need to select which ticker to tune on)
        ticker = self.ticker_input.text().split(',')[0].strip()
        self.worker = PipelineWorker(ticker, mode="tune")
        self.worker.log_signal.connect(self.log_area.append)
        self.worker.start()

    def run_train(self):
        ticker = self.ticker_input.text().split(',')[0].strip()
        self.log_area.append(f"Starting Training for {ticker}...")
        self.worker = PipelineWorker(ticker, mode="train")
        self.worker.log_signal.connect(self.log_area.append)
        self.worker.start()
