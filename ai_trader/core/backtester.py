import vectorbt as vbt
import pandas as pd
import numpy as np

class Backtester:
    def __init__(self):
        pass

    def run_backtest(self, df, preds, config):
        """
        df: Dataframe with 'Close' price
        preds: Binary array (1 for Buy, 0 for Hold/Sell)
        config: dictionary with fees, slippage, init_cash
        """
        price = df['Close']

        # Logic:
        # If preds[i] == 1 (Predicted UP), Enter Long
        # If preds[i] == 0 (Predicted DOWN), Exit Long
        # Basic Swing Strategy

        # Ensure preds align with index
        entries = pd.Series(preds == 1, index=df.index)
        exits = pd.Series(preds == 0, index=df.index)

        init_cash = config.get('initial_capital', 100000)
        fees = config.get('fees', 0.001)
        slippage = config.get('slippage', 0.001)

        pf = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            freq='1D',
            init_cash=init_cash,
            fees=fees,
            slippage=slippage
        )

        return pf

    def get_stats(self, pf):
        return pf.stats()

    def get_trades(self, pf):
        return pf.trades.records_readable
