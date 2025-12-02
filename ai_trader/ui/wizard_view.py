from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit,
                             QPushButton, QTextEdit, QHBoxLayout, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ai_trader.utils.workers import PipelineWorker

class WizardView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)

        # Header
        header = QLabel("Quick Start Auto-Analysis")
        header.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(header)

        desc = QLabel("Enter a ticker symbol below. The AI will automatically fetch data, tune hyperparameters, train a model, and generate a backtest report.")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(20)

        # Input
        input_layout = QHBoxLayout()
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("e.g., RELIANCE.NS or ^NSEI")
        self.ticker_input.setStyleSheet("padding: 10px; font-size: 16px;")
        input_layout.addWidget(self.ticker_input)

        self.btn_run = QPushButton("Run Auto-Analysis")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #27ae60; }
        """)
        self.btn_run.clicked.connect(self.start_pipeline)
        input_layout.addWidget(self.btn_run)

        layout.addLayout(input_layout)

        layout.addSpacing(20)

        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Log Output
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def start_pipeline(self):
        ticker = self.ticker_input.text().strip()
        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a valid ticker symbol.")
            return

        self.btn_run.setEnabled(False)
        self.log_area.clear()
        self.progress_bar.setValue(0)

        # Start Worker Thread
        self.worker = PipelineWorker(ticker, mode="wizard")
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def append_log(self, message):
        self.log_area.append(message)

    def on_finished(self, result):
        self.btn_run.setEnabled(True)
        if result.get("success"):
            QMessageBox.information(self, "Success", f"Analysis Complete!\nReport saved to: {result.get('report_path')}")
        else:
            QMessageBox.critical(self, "Error", f"Analysis Failed: {result.get('error')}")
