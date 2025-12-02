import sys
import os

# Mock PyQt to avoid display errors in headless environment
from unittest.mock import MagicMock
sys.modules["PyQt6.QtWidgets"] = MagicMock()
sys.modules["PyQt6.QtCore"] = MagicMock()

# Import core modules to verify syntax and imports
try:
    print("Verifying imports...")
    from ai_trader.config.settings import settings_manager
    from ai_trader.core.data_processor import DataProcessor
    from ai_trader.core.model_engine import ModelEngine
    from ai_trader.core.tuner import HyperparameterTuner
    from ai_trader.core.backtester import Backtester
    from ai_trader.core.reporting import ReportGenerator
    print("Imports successful.")

    # Check paths
    print(f"Data Dir: {settings_manager.get('data_dir')}") # might be None if I didn't expose it, checking logic

    # Instantiate classes
    dp = DataProcessor()
    me = ModelEngine()
    bt = Backtester()
    rg = ReportGenerator()
    print("Class instantiation successful.")

except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Runtime Error: {e}")
    sys.exit(1)

print("Verification passed.")
