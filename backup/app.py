import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
import datetime
import pandas as pd
import yfinance as yf
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.preprocessing import MinMaxScaler

# Matplotlib for Portfolio Analysis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import vectorbt as vbt

# Constants
SAVE_DIR = r"C:\Users\HP\Documents\py projects\Ai model1"
CONFIG_FILE = os.path.join(SAVE_DIR, "config.json")
MODEL_REGISTRY_FILE = os.path.join(SAVE_DIR, "models_registry.json")

# Default Settings
DEFAULT_SETTINGS = {
    "tickers": "RELIANCE.NS,^nsei,mon100.ns",
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_size": 64,
    "num_layers": 1,
    "seq_length": 5,  # Default sequence length
    "features": [
        "Returns", "RSI", "RSI_Open", "RSI_High", "RSI_Low", "MFI", "Rel_Vol", "ADX", "Dist_EMA200",
        "VIX_Close", "VIX_RSI", "VIX_RSI_Open", "VIX_RSI_High", "VIX_RSI_Low", "VIX_ADX", "VIX_EMA"
    ]
}


# --- Model Registry ---
class ModelRegistry:
    def __init__(self, registry_path):
        self.registry_path = registry_path
        self.registry = self.load_registry()

    def load_registry(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                return {}
        return {}

    def save_registry(self):
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=4)
        except Exception as e:
            print(f"Error saving registry: {e}")

    def register_model(self, name, description, params, features, files):
        self.registry[name] = {
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "parameters": params,
            "features": features,
            "files": files
        }
        self.save_registry()

    def get_model(self, name):
        return self.registry.get(name)

    def list_models(self):
        return list(self.registry.keys())


# --- PyTorch Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=1, num_classes=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class AITradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Swing Trader (Lightweight)")
        self.root.geometry("650x800") # Increased height

        # Ensure save directory exists
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        self.registry = ModelRegistry(MODEL_REGISTRY_FILE)
        self.setup_gui()
        self.load_config()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_gui(self):
        # --- Main Container ---
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # --- Tabs ---
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 1: Training
        self.tab_train = tk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="  Model Training  ")
        self.setup_training_tab()

        # Tab 2: Usage (Signals)
        self.tab_usage = tk.Frame(self.notebook)
        self.notebook.add(self.tab_usage, text="  Model Usage (Signals)  ")
        self.setup_usage_tab()

        # Log Window (Shared at bottom)
        log_frame = tk.LabelFrame(main_frame, text="System Log")
        log_frame.pack(fill="x", padx=10, pady=5, side="bottom")

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_training_tab(self):
        # Scrollable Canvas for Training Tab
        canvas = tk.Canvas(self.tab_train)
        scrollbar = ttk.Scrollbar(self.tab_train, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Content ---
        tk.Label(scroll_frame, text="Train New AI Models", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Instructions
        instr_frame = tk.LabelFrame(scroll_frame, text="Instructions")
        instr_frame.pack(fill="x", padx=20, pady=5)
        tk.Label(instr_frame, text="1. Enter Tickers & Index.\n2. Download Data.\n3. Configure & Train.", justify="left").pack(anchor="w", padx=10, pady=5)

        # Inputs
        input_frame = tk.LabelFrame(scroll_frame, text="Data Source")
        input_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(input_frame, text="Training Tickers (comma-separated):").pack(anchor="w", padx=5)
        self.ticker_entry = tk.Entry(input_frame)
        self.ticker_entry.insert(0, "RELIANCE.NS,TCS.NS,INFY.NS")
        self.ticker_entry.pack(fill="x", padx=5, pady=2)

        tk.Label(input_frame, text="Index Ticker:").pack(anchor="w", padx=5)
        self.index_entry = tk.Entry(input_frame)
        self.index_entry.insert(0, "^INDIAVIX")
        self.index_entry.pack(fill="x", padx=5, pady=2)

        self.btn_download = tk.Button(input_frame, text="Download & Label Data", command=self.run_data_prep_thread, bg="#e1f5fe")
        self.btn_download.pack(pady=10)

        # Configuration
        conf_frame = tk.LabelFrame(scroll_frame, text="Model Configuration")
        conf_frame.pack(fill="x", padx=20, pady=5)

        f1 = tk.Frame(conf_frame)
        f1.pack(fill="x", pady=2)
        tk.Label(f1, text="Model Name:").pack(side="left", padx=5)
        self.model_name_entry = tk.Entry(f1, width=20)
        self.model_name_entry.pack(side="left", padx=5)

        tk.Label(f1, text="Window Size:").pack(side="left", padx=5)
        self.window_size_entry = tk.Entry(f1, width=5)
        self.window_size_entry.insert(0, "5")
        self.window_size_entry.pack(side="left", padx=5)

        tk.Label(conf_frame, text="Description:").pack(anchor="w", padx=5)
        self.model_desc_entry = tk.Entry(conf_frame)
        self.model_desc_entry.pack(fill="x", padx=5, pady=2)

        self.train_mode = tk.StringVar(value="combined")
        # tk.Radiobutton(conf_frame, text="Combined Model", variable=self.train_mode, value="combined").pack(anchor="w", padx=5)

        tk.Button(conf_frame, text="Advanced Settings", command=self.open_settings).pack(pady=5)

        # Train Button
        self.btn_train = tk.Button(scroll_frame, text="TRAIN MODEL", command=self.run_train_thread, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"), height=2)
        self.btn_train.pack(pady=20, fill="x", padx=40)

    def setup_usage_tab(self):
        # Top Frame: Model Selection & Actions
        top_frame = tk.Frame(self.tab_usage)
        top_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(top_frame, text="Select Model:").pack(side="left")
        self.model_selector = ttk.Combobox(top_frame, state="readonly", width=30)
        self.model_selector.pack(side="left", padx=10)
        self.refresh_model_list()

        tk.Button(top_frame, text="Refresh Models", command=self.refresh_model_list).pack(side="left")

        # Middle Frame: Ticker Input
        mid_frame = tk.LabelFrame(self.tab_usage, text="Tickers to Scan (One per line or comma-separated)")
        mid_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.usage_ticker_text = scrolledtext.ScrolledText(mid_frame, height=5)
        self.usage_ticker_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.usage_ticker_text.insert("1.0", "RELIANCE.NS\nTCS.NS\nINFY.NS\nSBIN.NS")

        # Action Buttons
        btn_frame = tk.Frame(self.tab_usage)
        btn_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(btn_frame, text="SCAN & PREDICT SIGNALS", command=self.run_batch_scan_thread, bg="#2196F3", fg="white", font=("bold")).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(btn_frame, text="Export Results (CSV)", command=self.export_scan_results).pack(side="left", padx=5)

        # Results Table
        res_frame = tk.LabelFrame(self.tab_usage, text="Scan Results")
        res_frame.pack(fill="both", expand=True, padx=10, pady=5)

        cols = ("Ticker", "Signal", "Confidence", "Price", "Date")
        self.res_tree = ttk.Treeview(res_frame, columns=cols, show='headings')
        for col in cols:
            self.res_tree.heading(col, text=col)
            self.res_tree.column(col, width=100)

        self.res_tree.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(res_frame, orient="vertical", command=self.res_tree.yview)
        sb.pack(side="right", fill="y")
        self.res_tree.configure(yscrollcommand=sb.set)

        # Reports Section (Bottom)
        rep_frame = tk.LabelFrame(self.tab_usage, text="Analysis & Reports")
        rep_frame.pack(fill="x", padx=10, pady=5)

        self.btn_backtest = tk.Button(rep_frame, text="Backtest Report (Past 1 Year)", command=self.run_backtest_thread)
        self.btn_backtest.pack(side="left", padx=5, pady=5)

        tk.Button(rep_frame, text="VectorBT Analysis", command=self.generate_vectorbt_report).pack(side="left", padx=5, pady=5)
        tk.Button(rep_frame, text="Portfolio Plot", command=self.open_portfolio_analysis).pack(side="left", padx=5, pady=5)

    def update_log(self, message):
        """Updates the log window with a new message, keeping only the last 20 lines."""
        def _log():
            self.log_text.config(state='normal')
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")

            # Keep only last 20 lines
            num_lines = int(self.log_text.index('end-1c').split('.')[0])
            if num_lines > 20:
                self.log_text.delete('1.0', f'{num_lines - 20}.0')

            self.log_text.see(tk.END)
            self.log_text.config(state='disabled')

        self.root.after(0, _log)

    def refresh_model_list(self):
        models = self.registry.list_models()
        self.model_selector['values'] = models
        if models:
            self.model_selector.current(0)

    def load_config(self):
        self.settings = DEFAULT_SETTINGS.copy()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
            except Exception as e:
                print(f"Error loading config: {e}")

        # Update UI
        self.ticker_entry.delete(0, tk.END)
        self.ticker_entry.insert(0, self.settings.get("tickers", ""))

        usage_tickers = self.settings.get("usage_tickers", "RELIANCE.NS\nTCS.NS\nINFY.NS\nSBIN.NS")
        self.usage_ticker_text.delete("1.0", tk.END)
        self.usage_ticker_text.insert("1.0", usage_tickers)

    def save_config(self):
        self.settings["tickers"] = self.ticker_entry.get()
        self.settings["usage_tickers"] = self.usage_ticker_text.get("1.0", tk.END).strip()
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def on_close(self):
        self.save_config()
        self.root.destroy()

    def get_tickers(self):
        raw_text = self.ticker_entry.get()
        tickers = [t.strip() for t in raw_text.split(',') if t.strip()]
        return tickers

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("400x600")

        # Hyperparameters
        tk.Label(win, text="Hyperparameters", font=("bold")).pack(pady=5)

        params_frame = tk.Frame(win)
        params_frame.pack(pady=5, padx=10, fill="x")

        entries = {}
        row = 0
        for key in ["epochs", "batch_size", "learning_rate", "hidden_size", "num_layers"]:
            tk.Label(params_frame, text=key.replace("_", " ").title() + ":").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ent = tk.Entry(params_frame)
            ent.insert(0, str(self.settings.get(key, DEFAULT_SETTINGS[key])))
            ent.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            entries[key] = ent
            row += 1

        # Feature Selection
        tk.Label(win, text="Feature Selection", font=("bold")).pack(pady=10)

        features_frame = tk.Frame(win)
        features_frame.pack(pady=5, padx=10, fill="both", expand=True)

        all_features = DEFAULT_SETTINGS["features"]
        current_features = self.settings.get("features", all_features)

        feature_vars = {}
        for feat in all_features:
            var = tk.BooleanVar(value=feat in current_features)
            tk.Checkbutton(features_frame, text=feat, variable=var).pack(anchor="w")
            feature_vars[feat] = var

        def save_settings():
            try:
                # Update Settings Dict
                self.settings["epochs"] = int(entries["epochs"].get())
                self.settings["batch_size"] = int(entries["batch_size"].get())
                self.settings["learning_rate"] = float(entries["learning_rate"].get())
                self.settings["hidden_size"] = int(entries["hidden_size"].get())
                self.settings["num_layers"] = int(entries["num_layers"].get())

                selected_feats = [f for f, v in feature_vars.items() if v.get()]
                if not selected_feats:
                    messagebox.showerror("Error", "Select at least one feature.")
                    return
                self.settings["features"] = selected_feats

                self.save_config()
                win.destroy()
                messagebox.showinfo("Success", "Settings saved.")

            except ValueError:
                messagebox.showerror("Error", "Invalid input for numeric fields.")

        tk.Button(win, text="Save Settings", command=save_settings, bg="#4CAF50", fg="white").pack(pady=20, fill="x", padx=20)

    def run_batch_scan_thread(self):
        # Retrieve GUI values in Main Thread
        raw_text = self.usage_ticker_text.get("1.0", tk.END)
        selected_model_name = self.model_selector.get()
        index_ticker = self.index_entry.get().strip()

        threading.Thread(target=self.run_batch_scan, args=(raw_text, selected_model_name, index_ticker), daemon=True).start()

    def clear_results(self):
        self.root.after(0, lambda: [self.res_tree.delete(item) for item in self.res_tree.get_children()])

    def add_result_row(self, values):
        self.root.after(0, lambda: self.res_tree.insert("", "end", values=values))

    def run_batch_scan(self, raw_text, selected_model_name, index_ticker):
        """Logic for scanning multiple tickers in the Usage Tab."""
        # Split by newlines or commas
        tickers = [t.strip() for t in raw_text.replace('\n', ',').split(',') if t.strip()]

        if not tickers:
            self.update_log("Error: Please enter at least one ticker.")
            return

        if not selected_model_name:
            self.update_log("Error: Please select a model.")
            return

        self.update_log(f"Starting Batch Scan for {len(tickers)} tickers using {selected_model_name}...")

        # Clear previous results
        self.clear_results()

        # Load Model Config
        model_config = self.registry.get_model(selected_model_name)
        if not model_config:
            self.update_log("Error: Model config not found.")
            return

        try:
            # Extract Config
            seq_length = model_config['parameters']['seq_length']
            hidden_size = model_config['parameters']['hidden_size']
            num_layers = model_config['parameters']['num_layers']
            features_meta = model_config['features']
            features = features_meta['selected']

            stock_features = [f for f in features if not f.startswith("VIX_")]
            vix_features = [f for f in features if f.startswith("VIX_")]
            input_size = len(features)

            # Load Files
            model_path = os.path.join(SAVE_DIR, model_config['files']['model_path'])
            scaler_path = os.path.join(SAVE_DIR, model_config['files']['scaler_path'])

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                self.update_log("Error: Model files missing.")
                return

            # Load Scaler
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # Load Model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=3).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # Get Index Data (Once)
            if not index_ticker:
                 index_ticker = "^INDIAVIX" # Fallback

            vix_df = yf.download(index_ticker, period="1y", interval="1d", progress=False)

            if vix_df.empty:
                self.update_log("Error: Could not fetch Index Data.")
                return

            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            vix_df = vix_df[['Open', 'High', 'Low', 'Close']].copy()

            # Calculate Index Technicals (Same as training)
            vix_df['VIX_RSI'] = self.calculate_rsi(vix_df['Close'])
            vix_df['VIX_RSI_Open'] = self.calculate_rsi(vix_df['Open'])
            vix_df['VIX_RSI_High'] = self.calculate_rsi(vix_df['High'])
            vix_df['VIX_RSI_Low'] = self.calculate_rsi(vix_df['Low'])
            vix_df['VIX_ADX'] = self.calculate_adx(vix_df['High'], vix_df['Low'], vix_df['Close'])
            # Alias for backward compatibility
            vix_df['VIX_RSI_ADX'] = vix_df['VIX_ADX']
            vix_df['VIX_EMA'] = vix_df['Close'].ewm(span=20, adjust=False).mean()
            vix_df['VIX_Close'] = vix_df['Close']

            # Loop Tickers
            for ticker in tickers:
                try:
                    self.update_log(f"Scanning {ticker}...")
                    # Download recent data (enough for indicators)
                    df = yf.download(ticker, period="1y", interval="1d", progress=False)

                    if len(df) < 50: # Minimum data check
                        self.update_log(f"Skipping {ticker}: Not enough data.")
                        continue

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                    # Calculate Features
                    df['RSI'] = self.calculate_rsi(df['Close'])
                    df['RSI_Open'] = self.calculate_rsi(df['Open'])
                    df['RSI_High'] = self.calculate_rsi(df['High'])
                    df['RSI_Low'] = self.calculate_rsi(df['Low'])
                    df['MFI'] = self.calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
                    df['ADX'] = self.calculate_adx(df['High'], df['Low'], df['Close'])
                    df['R3'] = df['Close'] + (df['High'] - df['Low']) * 1.1 / 4
                    df['S3'] = df['Close'] - (df['High'] - df['Low']) * 1.1 / 4
                    df['Returns'] = df['Close'].pct_change()
                    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
                    df['Rel_Vol'] = df['Volume'] / df['Vol_SMA20']
                    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
                    df['Dist_EMA200'] = (df['Close'] - df['EMA200']) / df['EMA200']

                    # Merge Index
                    df = df.join(vix_df[vix_features], how='inner')
                    df.dropna(inplace=True)

                    if len(df) < seq_length:
                        continue

                    # Prepare Last Sequence
                    last_seq = df.iloc[-seq_length:]

                    # Check if all features exist
                    missing_cols = [c for c in stock_features if c not in df.columns]
                    if missing_cols:
                        self.update_log(f"Skipping {ticker}: Missing features {missing_cols}")
                        continue

                    scaled_stock = scaler.transform(last_seq[stock_features])
                    raw_vix = last_seq[vix_features].values
                    combined_features = np.hstack((scaled_stock, raw_vix))

                    X_input = combined_features.reshape(1, seq_length, input_size)
                    X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        logits = model(X_tensor)
                        probs = torch.softmax(logits, dim=1)

                    prob_sell = probs[0][0].item()
                    prob_hold = probs[0][1].item()
                    prob_buy = probs[0][2].item()

                    signal = "HOLD"
                    confidence = prob_hold

                    if prob_buy > 0.4 and prob_buy > prob_sell:
                        signal = "BUY"
                        confidence = prob_buy
                    elif prob_sell > 0.4 and prob_sell > prob_buy:
                        signal = "SELL"
                        confidence = prob_sell

                    last_price = df['Close'].iloc[-1]
                    date_str = datetime.datetime.now().strftime("%Y-%m-%d")

                    # Add to Table
                    self.add_result_row((ticker, signal, f"{confidence:.2f}", f"{last_price:.2f}", date_str))
                    self.update_log(f"{ticker}: {signal} ({confidence:.2f})")

                except Exception as e:
                    self.update_log(f"Error scanning {ticker}: {e}")
                    continue

            self.update_log("Batch Scan Complete.")

        except Exception as e:
            self.update_log(f"Critical Error in Batch Scan: {e}")
            print(e)

    def export_scan_results(self):
        rows = self.res_tree.get_children()
        if not rows:
            messagebox.showinfo("Info", "No results to export.")
            return

        data = []
        for row in rows:
            data.append(self.res_tree.item(row)['values'])

        df = pd.DataFrame(data, columns=["Ticker", "Signal", "Confidence", "Price", "Date"])
        path = os.path.join(SAVE_DIR, f"scan_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(path, index=False)
        messagebox.showinfo("Success", f"Saved to {path}")

    # --- Threading Wrappers ---
    def run_data_prep_thread(self):
        threading.Thread(target=self.prepare_data, daemon=True).start()

    def run_train_thread(self):
        threading.Thread(target=self.train_model, daemon=True).start()

    def run_predict_thread(self):
        threading.Thread(target=self.predict_signal, daemon=True).start()

    def run_backtest_thread(self):
        threading.Thread(target=self.backtest_predictions, daemon=True).start()

    # --- Helper Functions ---
    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_mfi(self, high, low, close, volume, period=14):
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        # Get positive and negative money flow
        delta = typical_price.diff()
        pos_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
        neg_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()

        mfi = 100 - (100 / (1 + pos_flow / neg_flow))
        return mfi

    def calculate_adx(self, high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='outer').max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        return adx

    def create_sequences(self, data, seq_length=15):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length), :-1] # All columns except Target
            y = data[i + seq_length, -1]      # Target column
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # --- Logic Functions ---
    def prepare_data(self):
        try:
            tickers = self.get_tickers()
            index_ticker = self.index_entry.get().strip()

            if not tickers:
                self.update_log("Error: No tickers provided.")
                return

            self.btn_download.config(state='disabled')

            # --- 1. Process Index Data ---
            self.update_log(f"Fetching Index Data ({index_ticker})...")
            vix_df = yf.download(index_ticker, period="5y", interval="1d", progress=False)

            if vix_df.empty:
                self.update_log(f"Error: Could not fetch {index_ticker}. Proceeding without it?")
                self.btn_download.config(state='normal')
                return

            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            vix_df = vix_df[['Open', 'High', 'Low', 'Close']].copy()

            # Calculate Index Technicals
            vix_df['VIX_RSI'] = self.calculate_rsi(vix_df['Close'])
            vix_df['VIX_RSI_Open'] = self.calculate_rsi(vix_df['Open'])
            vix_df['VIX_RSI_High'] = self.calculate_rsi(vix_df['High'])
            vix_df['VIX_RSI_Low'] = self.calculate_rsi(vix_df['Low'])
            vix_df['VIX_ADX'] = self.calculate_adx(vix_df['High'], vix_df['Low'], vix_df['Close'])
            # Alias for backward compatibility
            vix_df['VIX_RSI_ADX'] = vix_df['VIX_ADX']
            vix_df['VIX_EMA'] = vix_df['Close'].ewm(span=20, adjust=False).mean()
            vix_df['VIX_Close'] = vix_df['Close']

            # Keep only relevant columns
            vix_features = vix_df[['VIX_Close', 'VIX_RSI', 'VIX_RSI_Open', 'VIX_RSI_High', 'VIX_RSI_Low', 'VIX_ADX', 'VIX_RSI_ADX', 'VIX_EMA']].dropna()

            # --- 2. Process Stock Data ---
            for ticker in tickers:
                self.update_log(f"Processing {ticker}...")
                df = yf.download(ticker, period="5y", interval="1d", progress=False)

                if df.empty:
                    self.update_log(f"Warning: No data for {ticker}")
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                # Calculate RSI
                df['RSI'] = self.calculate_rsi(df['Close'])
                df['RSI_Open'] = self.calculate_rsi(df['Open'])
                df['RSI_High'] = self.calculate_rsi(df['High'])
                df['RSI_Low'] = self.calculate_rsi(df['Low'])

                # Calculate MFI
                df['MFI'] = self.calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])

                # Calculate ADX
                df['ADX'] = self.calculate_adx(df['High'], df['Low'], df['Close'])

                # Calculate Camarilla Pivots (R3, S3)
                df['R3'] = df['Close'] + (df['High'] - df['Low']) * 1.1 / 4
                df['S3'] = df['Close'] - (df['High'] - df['Low']) * 1.1 / 4

                # --- Feature Engineering (Relative Values) ---
                # 1. Returns (Percentage Change)
                df['Returns'] = df['Close'].pct_change()

                # 2. Relative Volume (Vol / SMA20)
                df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
                df['Rel_Vol'] = df['Volume'] / df['Vol_SMA20']

                # 3. Distance from EMA200
                df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
                df['Dist_EMA200'] = (df['Close'] - df['EMA200']) / df['EMA200']

                # --- Merge with Index Data ---
                # Inner join to ensure we have VIX data for every stock day
                df = df.join(vix_features, how='inner')

                # Create Target (T-2 Prediction)
                # Target = 1 if Returns(T+2) >= 1%
                # Target = -1 if Returns(T+2) <= -1%

                df['Future_Returns'] = df['Returns'].shift(-2)

                conditions = [
                    (df['Future_Returns'] >= 0.01),
                    (df['Future_Returns'] <= -0.01)
                ]
                choices = [1, -1]
                df['Target'] = np.select(conditions, choices, default=0)

                # Drop NaNs
                df.dropna(inplace=True)

                # Log Distribution
                buy_count = len(df[df['Target'] == 1])
                sell_count = len(df[df['Target'] == -1])
                hold_count = len(df[df['Target'] == 0])
                self.update_log(f"{ticker} -> Buy: {buy_count}, Sell: {sell_count}, Hold: {hold_count}")

                # Save to CSV
                save_path = os.path.join(SAVE_DIR, f"data_{ticker}.csv")
                df.to_csv(save_path)
                self.update_log(f"Saved {len(df)} rows to data_{ticker}.csv")

            self.update_log("Data Preparation Complete.")

        except Exception as e:
            self.update_log(f"Error in Data Prep: {str(e)}")
            print(e)
        finally:
            self.btn_download.config(state='normal')

    def train_model(self):
        self._run_train_logic()

    def _run_train_logic(self):
        tickers = self.get_tickers()
        mode = self.train_mode.get()
        model_name = self.model_name_entry.get().strip()
        model_desc = self.model_desc_entry.get().strip()

        try:
            seq_length = int(self.window_size_entry.get())
        except ValueError:
            self.update_log("Error: Invalid Window Size.")
            return

        if not model_name:
            self.update_log("Error: Please enter a Model Name.")
            return

        if not tickers:
            self.update_log("Error: No tickers provided.")
            return

        self.btn_train.config(state='disabled')
        try:
            # Load Settings
            epochs = self.settings.get("epochs", 100)
            batch_size = self.settings.get("batch_size", 32)
            lr = self.settings.get("learning_rate", 0.001)
            hidden_size = self.settings.get("hidden_size", 64)
            num_layers = self.settings.get("num_layers", 1)
            selected_features = self.settings.get("features", DEFAULT_SETTINGS["features"])

            # Identify Stock vs VIX features for scaling logic
            stock_features = [f for f in selected_features if not f.startswith("VIX_")]
            vix_features = [f for f in selected_features if f.startswith("VIX_")]

            feature_cols = stock_features + vix_features
            target_col = 'Target'

            # Define filenames based on Model Name
            model_filename = f"model_{model_name}.pth"
            scaler_filename = f"scaler_{model_name}.pkl"

            if mode == "combined":
                # Combined Model Logic
                combined_df_list = []
                for ticker in tickers:
                    file_path = os.path.join(SAVE_DIR, f"data_{ticker}.csv")
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        combined_df_list.append(df)

                if not combined_df_list:
                    self.update_log("Error: No data found.")
                    return

                full_df = pd.concat(combined_df_list, ignore_index=True)

                # Fit Scaler ONLY on Stock Features
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(full_df[stock_features])

                # Save Combined Scaler
                with open(os.path.join(SAVE_DIR, scaler_filename), "wb") as f:
                    pickle.dump(scaler, f)

                # Create Sequences
                X_all, y_all = [], []
                for ticker in tickers:
                    file_path = os.path.join(SAVE_DIR, f"data_{ticker}.csv")
                    if not os.path.exists(file_path): continue
                    df = pd.read_csv(file_path)

                    # Scale Stock Features
                    scaled_stock = scaler.transform(df[stock_features])
                    # Get Raw VIX Features
                    raw_vix = df[vix_features].values

                    # Combine
                    combined_features = np.hstack((scaled_stock, raw_vix))

                    target_values = df[target_col].values.reshape(-1, 1)
                    data_for_seq = np.hstack((combined_features, target_values))

                    X, y = self.create_sequences(data_for_seq, seq_length=seq_length)
                    if len(X) > 0:
                        X_all.append(X)
                        y_all.append(y)

                if not X_all:
                    self.update_log("Error: Not enough data.")
                    return

                X_train = np.concatenate(X_all)
                y_train = np.concatenate(y_all)

                self._train_pytorch_model(X_train, y_train, model_filename, hidden_size, num_layers, epochs, batch_size, lr, len(feature_cols))

                # Register Model
                params = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "seq_length": seq_length
                }
                features_meta = {
                    "selected": selected_features,
                    "stock_features": stock_features,
                    "vix_features": vix_features
                }
                files = {
                    "model_path": model_filename,
                    "scaler_path": scaler_filename
                }

                self.registry.register_model(model_name, model_desc, params, features_meta, files)
                self.update_log(f"Model '{model_name}' registered successfully.")

                # Refresh Dropdown
                self.root.after(0, self.refresh_model_list)

            else:
                self.update_log("Error: 'Separate Models' mode not fully supported with Registry yet. Use Combined.")
                # TODO: Implement separate models registry logic if needed.

            self.update_log("Training Complete.")

        except Exception as e:
            self.update_log(f"Error in Training: {str(e)}")
            print(e)
        finally:
            self.btn_train.config(state='normal')

    def _train_pytorch_model(self, X_train, y_train, model_filename, hidden_size, num_layers, epochs, batch_size, lr, input_size):
        # Map Targets: -1 (Sell) -> 0, 0 (Hold) -> 1, 1 (Buy) -> 2
        y_mapped = np.zeros_like(y_train)
        y_mapped[y_train == -1] = 0
        y_mapped[y_train == 0] = 1
        y_mapped[y_train == 1] = 2

        # Convert to Tensor
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_mapped, dtype=torch.long)

        # --- Oversampling Strategy ---
        unique, counts = np.unique(y_mapped, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # Weight for class i = 1 / count_i
        weights = [1.0 / class_counts.get(label, 1) for label in y_mapped]
        sample_weights = torch.DoubleTensor(weights)

        # WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        # Initialize Model
        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training Loop
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                self.update_log(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        # Save Model
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, model_filename))
        self.update_log(f"Saved {model_filename}")

        # --- Evaluation & Metrics Logging ---
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1).numpy()
            actual_targets = y_tensor.numpy()

            # Count Actuals
            act_sell = np.sum(actual_targets == 0)
            act_hold = np.sum(actual_targets == 1)
            act_buy = np.sum(actual_targets == 2)

            # Count Predicted
            pred_sell = np.sum(predictions == 0)
            pred_hold = np.sum(predictions == 1)
            pred_buy = np.sum(predictions == 2)

            metrics_msg = (
                f"\nModel: {model_filename}\n"
                f"ACTUAL    -> Buy: {act_buy}, Sell: {act_sell}, Hold: {act_hold}\n"
                f"PREDICTED -> Buy: {pred_buy}, Sell: {pred_sell}, Hold: {pred_hold}\n"
                f"{'-'*30}\n"
            )

            # Append to training_metrics.txt
            metrics_path = os.path.join(SAVE_DIR, "training_metrics.txt")
            with open(metrics_path, "a") as f:
                f.write(metrics_msg)

            self.update_log(f"Metrics saved to training_metrics.txt")

    def _run_prediction_logic(self, backtest=False):
        if backtest:
             # Use tickers from Usage Tab
             raw_text = self.usage_ticker_text.get("1.0", tk.END)
             tickers = [t.strip() for t in raw_text.replace('\n', ',').split(',') if t.strip()]
        else:
             # Use tickers from Training Tab (or wherever get_tickers points to)
             tickers = self.get_tickers()

        index_ticker = self.index_entry.get().strip()
        selected_model_name = self.model_selector.get()

        if not tickers:
            self.update_log("Error: No tickers provided.")
            return

        if not selected_model_name:
            self.update_log("Error: No model selected.")
            return

        # btn = self.btn_backtest if backtest else self.btn_predict # btn_predict removed/not used in this flow?
        if backtest:
             self.btn_backtest.config(state='disabled')

        recommendations = []

        try:
            # --- 1. Load Model Config from Registry ---
            model_entry = self.registry.get_model(selected_model_name)
            if not model_entry:
                self.update_log(f"Error: Model '{selected_model_name}' not found in registry.")
                return

            params = model_entry["parameters"]
            features_meta = model_entry["features"]
            files = model_entry["files"]

            seq_length = params.get("seq_length", 5)
            hidden_size = params.get("hidden_size", 64)
            num_layers = params.get("num_layers", 1)

            stock_features = features_meta["stock_features"]
            vix_features = features_meta["vix_features"]
            feature_cols = features_meta["selected"] # or stock + vix

            model_filename = files["model_path"]
            scaler_filename = files["scaler_path"]

            input_size = len(stock_features) + len(vix_features)

            # Check files exist
            if not os.path.exists(os.path.join(SAVE_DIR, model_filename)) or \
               not os.path.exists(os.path.join(SAVE_DIR, scaler_filename)):
                self.update_log("Error: Model files missing.")
                return

            # Load Scaler
            with open(os.path.join(SAVE_DIR, scaler_filename), "rb") as f:
                scaler = pickle.load(f)

            # Load Model
            model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=3)
            model.load_state_dict(torch.load(os.path.join(SAVE_DIR, model_filename)))
            model.eval()

            # --- 2. Process Index Data ---
            self.update_log(f"Fetching Index Data ({index_ticker})...")
            vix_df = yf.download(index_ticker, period="2y" if backtest else "1mo", interval="1d", progress=False)

            if vix_df.empty:
                self.update_log(f"Error: Could not fetch {index_ticker}.")
                return

            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            vix_df = vix_df[['Open', 'High', 'Low', 'Close']].copy()

            # Calculate Index Technicals
            vix_df['VIX_RSI'] = self.calculate_rsi(vix_df['Close'])
            vix_df['VIX_RSI_Open'] = self.calculate_rsi(vix_df['Open'])
            vix_df['VIX_RSI_High'] = self.calculate_rsi(vix_df['High'])
            vix_df['VIX_RSI_Low'] = self.calculate_rsi(vix_df['Low'])
            vix_df['VIX_ADX'] = self.calculate_adx(vix_df['High'], vix_df['Low'], vix_df['Close'])
            # Alias for backward compatibility
            vix_df['VIX_RSI_ADX'] = vix_df['VIX_ADX']
            vix_df['VIX_EMA'] = vix_df['Close'].ewm(span=20, adjust=False).mean()
            vix_df['VIX_Close'] = vix_df['Close']

            # Filter VIX data based on config
            vix_data = vix_df[vix_features].dropna()

            # --- 3. Process Each Ticker ---
            for ticker in tickers:
                self.update_log(f"Predicting for {ticker}...")

                # Download Stock Data
                period = "2y" if backtest else "6mo" # Need enough for indicators + seq_length
                df = yf.download(ticker, period=period, interval="1d", progress=False)

                if df.empty: continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                # Calculate Indicators
                df['RSI'] = self.calculate_rsi(df['Close'])
                df['RSI_Open'] = self.calculate_rsi(df['Open'])
                df['RSI_High'] = self.calculate_rsi(df['High'])
                df['RSI_Low'] = self.calculate_rsi(df['Low'])
                df['MFI'] = self.calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
                df['ADX'] = self.calculate_adx(df['High'], df['Low'], df['Close'])

                df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
                df['Rel_Vol'] = df['Volume'] / df['Vol_SMA20']

                df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
                df['Dist_EMA200'] = (df['Close'] - df['EMA200']) / df['EMA200']

                df['Returns'] = df['Close'].pct_change()

                # Merge with VIX
                df = df.join(vix_data, how='inner')
                df.dropna(inplace=True)

                if len(df) < seq_length + 1:
                    self.update_log(f"Not enough data for {ticker}")
                    continue

                if backtest:
                    # Backtest Logic: Predict for every day in the past year (approx)
                    # We need to create sequences for the whole period

                    # Scale Stock Features
                    scaled_stock = scaler.transform(df[stock_features])
                    raw_vix = df[vix_features].values
                    combined_features = np.hstack((scaled_stock, raw_vix))

                    # Create sequences manually
                    X_list = []
                    valid_indices = []

                    for i in range(len(combined_features) - seq_length):
                        seq = combined_features[i : i+seq_length]
                        X_list.append(seq)
                        valid_indices.append(df.index[i+seq_length])

                    if not X_list: continue

                    X_arr = np.array(X_list)
                    X_tensor = torch.tensor(X_arr, dtype=torch.float32)

                    with torch.no_grad():
                        logits = model(X_tensor)
                        probs = torch.softmax(logits, dim=1)
                        # preds = torch.argmax(probs, dim=1).numpy() # Not used directly anymore

                    # Process predictions for report
                    for idx, prob_row in enumerate(probs):
                        prob_sell = prob_row[0].item()
                        prob_hold = prob_row[1].item()
                        prob_buy = prob_row[2].item()

                        signal = "HOLD"
                        confidence = prob_hold
                        threshold = 0.4

                        if prob_buy > threshold and prob_buy > prob_sell:
                            signal = "BUY"
                            confidence = prob_buy
                        elif prob_sell > threshold and prob_sell > prob_buy:
                            signal = "SELL"
                            confidence = prob_sell

                        # Logic for Saving Recommendation
                        current_date_idx = valid_indices[idx]

                        # Find index in original df
                        try:
                            df_idx = df.index.get_loc(current_date_idx)

                            if df_idx + 2 < len(df):
                                entry_price = df['Close'].iloc[df_idx]
                                exit_price = df['Close'].iloc[df_idx+2]
                                exit_date = df.index[df_idx+2]

                                if signal == "BUY":
                                    actual_return = (exit_price - entry_price) / entry_price
                                elif signal == "SELL":
                                    actual_return = -1 * (exit_price - entry_price) / entry_price
                                else:
                                    actual_return = 0.0

                                if signal in ["BUY", "SELL"] and confidence >= 0.7:
                                    recommendations.append({
                                        "Ticker": ticker,
                                        "Date": current_date_idx.strftime("%Y-%m-%d"),
                                        "Prediction": signal,
                                        "Probability": f"{confidence:.4f}",
                                        "Close_Price": f"{entry_price:.2f}",
                                        "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                                        "Exit_Price": f"{exit_price:.2f}",
                                        "Actual_Return": actual_return,
                                        "Model_Name": selected_model_name
                                    })
                        except Exception as e:
                            print(f"Error processing index {current_date_idx}: {e}")
                            continue

                else:
                    # Prediction for Tomorrow (using last available sequence)
                    last_seq_data = df.iloc[-seq_length:]

                    if len(last_seq_data) < seq_length:
                        self.update_log(f"Not enough recent data for {ticker}")
                        continue

                    # Scale
                    scaled_stock = scaler.transform(last_seq_data[stock_features])
                    raw_vix = last_seq_data[vix_features].values
                    combined_features = np.hstack((scaled_stock, raw_vix))

                    # Shape: (1, seq_length, input_size)
                    X_input = combined_features.reshape(1, seq_length, input_size)
                    X_tensor = torch.tensor(X_input, dtype=torch.float32)

                    with torch.no_grad():
                        logits = model(X_tensor)
                        probs = torch.softmax(logits, dim=1)
                        pred_class = torch.argmax(probs, dim=1).item()

                    signal = "HOLD"
                    if pred_class == 0: signal = "SELL"
                    elif pred_class == 2: signal = "BUY"

                    prob_buy = probs[0][2].item()
                    prob_sell = probs[0][0].item()

                    msg = f"{ticker}: {signal} (Buy Prob: {prob_buy:.2f}, Sell Prob: {prob_sell:.2f})"
                    self.update_log(msg)

                    # Add to recommendations list for CSV
                    confidence = prob_buy if signal == "BUY" else (prob_sell if signal == "SELL" else probs[0][1].item())
                    last_close = df['Close'].iloc[-1]

                    if signal in ["BUY", "SELL"] and confidence >= 0.7:
                         recommendations.append({
                            "Ticker": ticker,
                            "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                            "Prediction": signal,
                            "Probability": f"{confidence:.4f}",
                            "Close_Price": f"{last_close:.2f}",
                            "Exit_Date": "T+2",
                            "Exit_Price": "N/A",
                            "Actual_Return": 0.0,
                            "Model_Name": selected_model_name
                        })

            # Save Recommendations
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                rec_path = os.path.join(SAVE_DIR, "recommendations.csv")

                if backtest:
                     rec_df.to_csv(rec_path, index=False)
                     self.update_log(f"Saved {len(recommendations)} rows to recommendations.csv")
                else:
                    if os.path.exists(rec_path):
                        rec_df.to_csv(rec_path, mode='a', header=False, index=False)
                    else:
                        rec_df.to_csv(rec_path, index=False)
                    # Show info box for predictions
                    display_msgs = [f"{r['Ticker']}: {r['Prediction']} ({r['Probability']})" for r in recommendations]
                    messagebox.showinfo("Predictions", "\n".join(display_msgs))
            else:
                self.update_log("No high-confidence trades found.")

        except Exception as e:
            self.update_log(f"Error in Prediction: {str(e)}")
            print(e)
        finally:
            btn.config(state='normal')



    def predict_signal(self):
        self.update_log("Starting Prediction (Tomorrow)...")
        self._run_prediction_logic(backtest=False)

    def backtest_predictions(self):
        self.update_log("Starting Backtest (Past 1 Year)...")
        self._run_prediction_logic(backtest=True)

    def generate_backtest_report(self):
        rec_path = os.path.join(SAVE_DIR, "recommendations.csv")
        if not os.path.exists(rec_path):
            messagebox.showerror("Error", "recommendations.csv not found. Run Backtest first.")
            return

        try:
            df = pd.read_csv(rec_path)
            required_cols = ['Ticker', 'Prediction', 'Probability', 'Actual_Return', 'Date', 'Close_Price']
            if not all(col in df.columns for col in required_cols):
                 messagebox.showerror("Error", "CSV missing columns. Please re-run 'Generate Past 1 Year Recommendations'.")
                 return

            # Filter High Confidence
            high_conf = df[df['Probability'] >= 0.7].copy()

            if high_conf.empty:
                messagebox.showinfo("Report", "No trades found with Probability >= 0.7")
                return

            # Calculate Profit
            high_conf['Profit'] = 0.0
            high_conf.loc[high_conf['Prediction'] == 'BUY', 'Profit'] = high_conf['Actual_Return']
            high_conf.loc[high_conf['Prediction'] == 'SELL', 'Profit'] = -1 * high_conf['Actual_Return']

            # Filter for BUY/SELL only
            trades = high_conf[high_conf['Prediction'].isin(['BUY', 'SELL'])].copy()

            if trades.empty:
                messagebox.showinfo("Report", "No BUY/SELL trades found with Probability >= 0.7")
                return

            # Create Excel Writer
            report_path = os.path.join(SAVE_DIR, "backtest_report.xlsx")
            try:
                with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                    # Get unique tickers
                    tickers = trades['Ticker'].unique()

                    for ticker in tickers:
                        ticker_trades = trades[trades['Ticker'] == ticker]

                        # --- Combined Stats ---
                        total_trades = len(ticker_trades)
                        profitable_trades = len(ticker_trades[ticker_trades['Profit'] > 0])
                        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
                        total_profit = ticker_trades['Profit'].sum() * 100

                        # --- Buy Stats ---
                        buy_trades = ticker_trades[ticker_trades['Prediction'] == 'BUY']
                        buy_total = len(buy_trades)
                        buy_profitable = len(buy_trades[buy_trades['Profit'] > 0])
                        buy_win_rate = (buy_profitable / buy_total) * 100 if buy_total > 0 else 0
                        buy_profit = buy_trades['Profit'].sum() * 100

                        # --- Sell Stats ---
                        sell_trades = ticker_trades[ticker_trades['Prediction'] == 'SELL']
                        sell_total = len(sell_trades)
                        sell_profitable = len(sell_trades[sell_trades['Profit'] > 0])
                        sell_win_rate = (sell_profitable / sell_total) * 100 if sell_total > 0 else 0
                        sell_profit = sell_trades['Profit'].sum() * 100

                        # Create Summary DataFrame
                        summary_data = {
                            "Metric": [
                                "Total Trades", "Combined Win Rate (%)", "Combined Profit (%)",
                                "---",
                                "Total BUY Trades", "BUY Win Rate (%)", "BUY Profit (%)",
                                "---",
                                "Total SELL Trades", "SELL Win Rate (%)", "SELL Profit (%)"
                            ],
                            "Value": [
                                total_trades, f"{win_rate:.2f}", f"{total_profit:.2f}",
                                "",
                                buy_total, f"{buy_win_rate:.2f}", f"{buy_profit:.2f}",
                                "",
                                sell_total, f"{sell_win_rate:.2f}", f"{sell_profit:.2f}"
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)

                        # Write Summary
                        summary_df.to_excel(writer, sheet_name=ticker, startrow=0, index=False)

                        # Write Details (Leave a gap)
                        ticker_trades.drop(columns=['Ticker'], inplace=True) # Redundant in sheet
                        ticker_trades.to_excel(writer, sheet_name=ticker, startrow=14, index=False)

                messagebox.showinfo("Success", f"Detailed report saved to:\n{report_path}")

            except ImportError:
                messagebox.showerror("Error", "openpyxl library is missing. Please install it to generate Excel reports.\nRun: pip install openpyxl")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to write Excel: {str(e)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")

    def generate_vectorbt_report(self):
        rec_path = os.path.join(SAVE_DIR, "recommendations.csv")
        if not os.path.exists(rec_path):
            messagebox.showerror("Error", "recommendations.csv not found. Run Backtest first.")
            return

        try:
            df = pd.read_csv(rec_path)
            # Ensure required columns
            required = ['Ticker', 'Date', 'Prediction', 'Exit_Date']
            if not all(col in df.columns for col in required):
                messagebox.showerror("Error", "CSV missing columns (Exit_Date). Re-run Backtest.")
                return

            # Filter High Confidence and Valid Signals
            trades = df[(df['Prediction'].isin(['BUY', 'SELL'])) & (df['Probability'] >= 0.7)].copy()

            if trades.empty:
                messagebox.showinfo("Info", "No trades found.")
                return

            unique_tickers = trades['Ticker'].unique().tolist()

            # --- Ticker Selection Dialog ---
            selection_win = tk.Toplevel(self.root)
            selection_win.title("Select Tickers")
            selection_win.geometry("300x400")

            tk.Label(selection_win, text="Select Tickers for Report:", font=("Helvetica", 10, "bold")).pack(pady=10)

            # Scrollable Frame for Checkboxes
            canvas = tk.Canvas(selection_win)
            scrollbar = ttk.Scrollbar(selection_win, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True, padx=5)
            scrollbar.pack(side="right", fill="y")

            ticker_vars = {}
            for ticker in unique_tickers:
                var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(scrollable_frame, text=ticker, variable=var)
                chk.pack(anchor="w", padx=10)
                ticker_vars[ticker] = var

            def proceed():
                selected_tickers = [t for t, v in ticker_vars.items() if v.get()]
                if not selected_tickers:
                    messagebox.showwarning("Warning", "No tickers selected.")
                    return
                selection_win.destroy()
                self._run_vectorbt_analysis(trades, selected_tickers)

            tk.Button(selection_win, text="Generate Report", command=proceed, bg="#4CAF50", fg="white").pack(pady=10, fill="x", padx=20)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load recommendations: {str(e)}")
            print(e)

    def _run_vectorbt_analysis(self, all_trades, selected_tickers):
        try:
            self.portfolios = {} # Store portfolios for plotting
            full_report = ""

            self.update_log(f"VectorBT: Analyzing {len(selected_tickers)} tickers...")

            for ticker in selected_tickers:
                ticker_trades = all_trades[all_trades['Ticker'] == ticker]
                if ticker_trades.empty: continue

                self.update_log(f"Processing {ticker}...")

                # Determine Date Range
                start_date = pd.to_datetime(ticker_trades['Date'].min())
                end_date = pd.to_datetime(ticker_trades['Exit_Date'].max())

                # Buffer dates
                start_date -= datetime.timedelta(days=10)
                end_date += datetime.timedelta(days=10)

                # Download OHLC Data for THIS ticker only
                data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)

                if data.empty:
                    full_report += f"--- {ticker} ---\nError: No data downloaded.\n\n"
                    continue

                # Handle MultiIndex if present
                if isinstance(data.columns, pd.MultiIndex):
                     # If single ticker download returns MultiIndex, it usually has Ticker as level 1
                     # But yfinance recent versions might behave differently.
                     # Safe bet: try to drop level if it exists or just take Close
                     try:
                         close_price = data['Close']
                         if isinstance(close_price, pd.DataFrame):
                             # If it's a DataFrame, it might have columns as Tickers
                             if ticker in close_price.columns:
                                 close_price = close_price[ticker]
                             else:
                                 # Fallback: take first column
                                 close_price = close_price.iloc[:, 0]
                     except:
                         close_price = data['Close']
                else:
                    close_price = data['Close']

                # Ensure Series
                if isinstance(close_price, pd.DataFrame):
                     close_price = close_price.iloc[:, 0]

                # Align Index
                close_price.index = pd.to_datetime(close_price.index).tz_localize(None)

                # Build Signals
                entries = pd.Series(False, index=close_price.index)
                exits = pd.Series(False, index=close_price.index)
                short_entries = pd.Series(False, index=close_price.index)
                short_exits = pd.Series(False, index=close_price.index)

                for _, row in ticker_trades.iterrows():
                    entry_date = pd.to_datetime(row['Date'])
                    exit_date = pd.to_datetime(row['Exit_Date'])
                    signal = row['Prediction']

                    try:
                        # Find nearest valid dates
                        entry_idx = close_price.index.get_indexer([entry_date], method='nearest')[0]
                        exit_idx = close_price.index.get_indexer([exit_date], method='nearest')[0]

                        actual_entry_date = close_price.index[entry_idx]
                        actual_exit_date = close_price.index[exit_idx]

                        # Prevent same-day entry/exit if logic requires holding
                        if actual_entry_date == actual_exit_date:
                            continue

                        if signal == 'BUY':
                            entries.at[actual_entry_date] = True
                            exits.at[actual_exit_date] = True
                        elif signal == 'SELL':
                            short_entries.at[actual_entry_date] = True
                            short_exits.at[actual_exit_date] = True

                    except Exception as e:
                        print(f"Date mapping error for {ticker}: {e}")
                        continue

                # Run VectorBT Portfolio for THIS ticker
                pf = vbt.Portfolio.from_signals(
                    close=close_price,
                    entries=entries,
                    exits=exits,
                    short_entries=short_entries,
                    short_exits=short_exits,
                    freq='1D',
                    init_cash=100000,
                    fees=0.001,
                    slippage=0.001
                )

                self.portfolios[ticker] = pf

                # Append Stats
                full_report += f"=== {ticker} REPORT ===\n"
                stats = pf.stats()
                full_report += stats.to_string()

                # Separate Long/Short Stats
                full_report += "\n\n--- LONG TRADES ONLY ---\n"
                try:
                    long_stats = pf.stats(direction='longonly')
                    full_report += long_stats.to_string()
                except:
                    full_report += "No long trades or error calculating."

                full_report += "\n\n--- SHORT TRADES ONLY ---\n"
                try:
                    short_stats = pf.stats(direction='shortonly')
                    full_report += short_stats.to_string()
                except:
                    full_report += "No short trades or error calculating."

                full_report += "\n\n" + "-"*40 + "\n\n"

            self.update_log("VectorBT: Analysis Complete.")
            self._show_vectorbt_results(full_report)

        except Exception as e:
            self.update_log(f"VectorBT Error: {str(e)}")
            print(e)
            messagebox.showerror("Error", f"VectorBT Analysis Failed: {str(e)}")

    def _show_vectorbt_results(self, report_text):
        top = tk.Toplevel(self.root)
        top.title("VectorBT Detailed Report")
        top.geometry("700x800")

        # Controls Frame
        ctrl_frame = tk.Frame(top)
        ctrl_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(ctrl_frame, text="Select Ticker to Plot:").pack(side="left")

        ticker_list = list(self.portfolios.keys())
        if not ticker_list:
            ticker_list = ["No Data"]

        self.plot_var = tk.StringVar(value=ticker_list[0])
        combo = ttk.Combobox(ctrl_frame, textvariable=self.plot_var, values=ticker_list, state="readonly")
        combo.pack(side="left", padx=5)

        def plot_selected():
            ticker = self.plot_var.get()
            if ticker in self.portfolios:
                self.portfolios[ticker].plot().show()
            else:
                messagebox.showwarning("Warning", "No portfolio data for selected ticker.")

        tk.Button(ctrl_frame, text="Plot Portfolio", command=plot_selected).pack(side="left", padx=10)

        # Text Area
        text_area = scrolledtext.ScrolledText(top, font=("Consolas", 10))
        text_area.pack(fill="both", expand=True, padx=10, pady=10)
        text_area.insert(tk.END, report_text)

    def open_portfolio_analysis(self):
        rec_path = os.path.join(SAVE_DIR, "recommendations.csv")
        if not os.path.exists(rec_path):
            messagebox.showerror("Error", "recommendations.csv not found. Run Backtest first.")
            return

        try:
            df = pd.read_csv(rec_path)
            tickers = sorted(df['Ticker'].unique())

            # Create Popup
            top = tk.Toplevel(self.root)
            top.title("Portfolio Analysis")
            top.geometry("400x600")

            tk.Label(top, text="Select Tickers for Portfolio:", font=("Helvetica", 12, "bold")).pack(pady=10)

            # Select All Checkbox
            self.select_all_var = tk.BooleanVar(value=True)
            def toggle_all():
                state = self.select_all_var.get()
                for var in self.ticker_vars.values():
                    var.set(state)

            tk.Checkbutton(top, text="Select All", variable=self.select_all_var, command=toggle_all).pack(anchor="w", padx=20)

            # Scrollable Frame for Checkboxes
            canvas = tk.Canvas(top)
            scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
            scroll_frame = tk.Frame(canvas)

            scroll_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True, padx=10)
            scrollbar.pack(side="right", fill="y")

            # Checkboxes
            self.ticker_vars = {}
            for ticker in tickers:
                var = tk.BooleanVar(value=True) # Default select all
                chk = tk.Checkbutton(scroll_frame, text=ticker, variable=var)
                chk.pack(anchor="w")
                self.ticker_vars[ticker] = var

            # Plot Button
            tk.Button(top, text="Plot Cumulative Returns", command=lambda: self.plot_portfolio(df), bg="#4CAF50", fg="white").pack(pady=10, fill="x", padx=20)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open portfolio analysis: {str(e)}")

    def plot_portfolio(self, df):
        selected_tickers = [t for t, var in self.ticker_vars.items() if var.get()]
        if not selected_tickers:
            messagebox.showwarning("Warning", "Please select at least one ticker.")
            return

        try:
            # Filter Data
            mask = (df['Ticker'].isin(selected_tickers)) & \
                   (df['Prediction'].isin(['BUY', 'SELL'])) & \
                   (df['Probability'] >= 0.7)

            filtered_df = df[mask].copy()

            if filtered_df.empty:
                messagebox.showinfo("Info", "No high-confidence trades found for selected tickers.")
                return

            # Calculate Profit
            filtered_df['Profit'] = 0.0
            filtered_df.loc[filtered_df['Prediction'] == 'BUY', 'Profit'] = filtered_df['Actual_Return']
            filtered_df.loc[filtered_df['Prediction'] == 'SELL', 'Profit'] = -1 * filtered_df['Actual_Return']

            # Convert Date
            filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
            filtered_df.sort_values('Date', inplace=True)

            # --- Cumulative Returns ---
            # 1. Total
            total_daily = filtered_df.groupby('Date')['Profit'].sum()
            total_cum = total_daily.cumsum() * 100

            # 2. Buy Only
            buy_df = filtered_df[filtered_df['Prediction'] == 'BUY']
            buy_daily = buy_df.groupby('Date')['Profit'].sum()
            # Reindex to match total dates to handle gaps, fill with 0
            buy_daily = buy_daily.reindex(total_daily.index, fill_value=0)
            buy_cum = buy_daily.cumsum() * 100

            # 3. Sell Only
            sell_df = filtered_df[filtered_df['Prediction'] == 'SELL']
            sell_daily = sell_df.groupby('Date')['Profit'].sum()
            sell_daily = sell_daily.reindex(total_daily.index, fill_value=0)
            sell_cum = sell_daily.cumsum() * 100

            # --- Plotting ---
            plt.figure(figsize=(10, 6))
            plt.plot(total_cum.index, total_cum, label='Total Portfolio', color='blue', linewidth=2)
            plt.plot(buy_cum.index, buy_cum, label='Buy Trades', color='green', linestyle='--')
            plt.plot(sell_cum.index, sell_cum, label='Sell Trades', color='red', linestyle='--')

            plt.title(f"Cumulative Portfolio Returns ({len(selected_tickers)} Tickers)")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (%)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot portfolio: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AITradingApp(root)
    root.mainloop()
