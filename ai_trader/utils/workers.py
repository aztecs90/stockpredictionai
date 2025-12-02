from PyQt6.QtCore import QThread, pyqtSignal
from ai_trader.core.data_processor import DataProcessor
from ai_trader.core.model_engine import ModelEngine
from ai_trader.core.tuner import HyperparameterTuner
from ai_trader.core.backtester import Backtester
from ai_trader.core.reporting import ReportGenerator
from ai_trader.config.settings import settings_manager
import traceback

class PipelineWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, ticker, mode="wizard"):
        super().__init__()
        self.ticker = ticker
        self.mode = mode # 'wizard', 'tune', 'train'

    def run(self):
        try:
            self.log_signal.emit(f"Starting {self.mode} process for {self.ticker}...")

            # 1. Data Fetching
            self.log_signal.emit("Fetching and processing data...")
            dp = DataProcessor()
            feat_config = settings_manager.get("features", {}).get("enabled", [])
            use_vix = settings_manager.get("features", {}).get("use_vix", True)

            df = dp.get_processed_data(self.ticker, start_date="2020-01-01", end_date=None,
                                       features_config=feat_config, use_vix=use_vix)

            if df is None or df.empty:
                raise ValueError("Data fetch failed or empty.")

            self.progress_signal.emit(20)

            # 2. Tuning (Optional / Wizard)
            engine = ModelEngine()
            train_config = settings_manager.get("training", {})

            if self.mode in ["wizard", "tune"]:
                self.log_signal.emit("Running Hyperparameter Tuning...")
                tuner = HyperparameterTuner(df, train_config)
                # Simple grid for demo
                grid = {
                    'learning_rate': [0.01, 0.001],
                    'hidden_size': [32, 64],
                    'tuning_epochs': [5]
                }
                best_params, best_acc = tuner.grid_search(grid, callback=lambda msg: self.log_signal.emit(msg))
                self.log_signal.emit(f"Best Accuracy found: {best_acc:.4f}")
                train_config.update(best_params)

            self.progress_signal.emit(50)

            if self.mode == "tune":
                self.finished_signal.emit({"success": True, "params": train_config})
                return

            # 3. Training
            self.log_signal.emit("Training Model...")
            acc = engine.train(df, f"model_{self.ticker}", train_config,
                               callback=lambda ep, loss: self.log_signal.emit(f"Epoch {ep}: Loss {loss:.4f}"))
            self.log_signal.emit(f"Final Test Accuracy: {acc:.4f}")

            self.progress_signal.emit(80)

            # 4. Backtesting
            self.log_signal.emit("Running Backtest Simulation...")
            preds, probs = engine.predict(df, train_config)

            bt = Backtester()
            bt_config = settings_manager.get("backtest", {})
            pf = bt.run_backtest(df, preds, bt_config)

            stats = bt.get_stats(pf)
            self.log_signal.emit("Backtest Complete.")

            # 5. Reporting
            self.log_signal.emit("Generating PDF Report...")
            rg = ReportGenerator()
            report_path = rg.generate_pdf(self.ticker, stats, pf, filename=f"Report_{self.ticker}.pdf")

            self.progress_signal.emit(100)
            self.finished_signal.emit({"success": True, "report_path": report_path})

        except Exception as e:
            traceback.print_exc()
            self.log_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit({"success": False, "error": str(e)})
