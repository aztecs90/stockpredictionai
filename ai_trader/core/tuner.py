import random
import itertools
from ai_trader.core.model_engine import ModelEngine
import numpy as np

class HyperparameterTuner:
    def __init__(self, df, base_config):
        self.df = df
        self.base_config = base_config
        self.engine = ModelEngine()

    def grid_search(self, param_grid, max_trials=10, callback=None):
        """
        Performs a random search over the parameter grid.
        param_grid: dict of lists, e.g. {'learning_rate': [0.01, 0.001], 'hidden_size': [32, 64]}
        max_trials: maximum number of combinations to try.
        """
        keys = param_grid.keys()
        combinations = list(itertools.product(*param_grid.values()))
        random.shuffle(combinations)

        best_acc = 0
        best_params = self.base_config.copy()

        trials = min(max_trials, len(combinations))

        for i in range(trials):
            # Construct config for this trial
            trial_params = dict(zip(keys, combinations[i]))
            current_config = self.base_config.copy()
            current_config.update(trial_params)

            if callback:
                callback(f"Trial {i+1}/{trials}: Testing {trial_params}...")

            # Train a temporary model
            try:
                # Use fewer epochs for tuning to save time, unless specified
                tune_config = current_config.copy()
                if 'tuning_epochs' in param_grid:
                     tune_config['epochs'] = param_grid['tuning_epochs'] # Allow overriding
                else:
                     tune_config['epochs'] = max(5, int(current_config['epochs'] / 2)) # 50% epochs for speed

                acc = self.engine.train(self.df, "temp_tuner", tune_config)

                if callback:
                    callback(f"Trial {i+1} Result: Accuracy {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_params = current_config

            except Exception as e:
                if callback:
                    callback(f"Trial {i+1} Failed: {e}")

        return best_params, best_acc
