import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, BooleanVar
import json
import base64
import numpy as np
from bs4 import BeautifulSoup  # You'll need to: pip install beautifulsoup4
import argparse
from typing import Tuple
from pathlib import Path

def get_server_config(config_file: str) -> dict:
    """Read server configuration from a FreqTrade config file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            api_config = config.get('api_server', {})
            if api_config.get('enabled', False):
                server_config = {
                    "url": f"http://{api_config.get('listen_ip_address', '127.0.0.1')}:{api_config.get('listen_port', 8080)}",
                    "username": api_config.get('username', 'freqtrader'),
                    "password": api_config.get('password', 'password'),
                    "config_file": config_file
                }
                print(f"Loaded config for {config_file} with port {api_config.get('listen_port', 8080)}")  # Debug line
                return server_config
    except Exception as e:
        print(f"Error reading config file {config_file}: {e}")
    return None

def control_trading(self, action, balance_ratio=None):
    """Control trading across all FreqTrade instances loaded in the config list"""
    success = True
    
    # Get list of config files from the GUI config list
    config_files = list(self.config_list.get(0, tk.END))
    if not config_files:
        print("No config files loaded in the list")
        return False
    
    for config_file in config_files:
        server_config = get_server_config(config_file)
        if not server_config:
            print(f"Skipping invalid config file: {config_file}")
            success = False
            continue
            
        try:
            credentials = f"{server_config['username']}:{server_config['password']}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/json'
            }
            
            if action == 'stop':
                url = f"{server_config['url']}/api/v1/stop"
            elif action == 'resume':
                url = f"{server_config['url']}/api/v1/start"
            elif action == 'stopbuy':
                url = f"{server_config['url']}/api/v1/stopbuy"
            elif action == 'reload':
                url = f"{server_config['url']}/api/v1/reload_config"
            elif action == 'adjust':
                # Update config file with new ratio
                if balance_ratio is not None:
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        # Ensure 2 decimal places when writing to config
                        config['tradable_balance_ratio'] = float(f"{balance_ratio:.2f}")
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=4)
                        print(f"Updated ratio in {config_file}")
                    except Exception as e:
                        print(f"Error updating config file {config_file}: {e}")
                        success = False
                        continue
                
                # Reload config
                url = f"{server_config['url']}/api/v1/reload_config"
            else:
                print(f"Unknown action: {action}")
                return False
            
            response = requests.post(url, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"Successfully executed {action} on {server_config['url']} ({config_file})")
            else:
                print(f"Failed to execute {action} on {server_config['url']} ({config_file}): {response.status_code}")
                success = False
        except Exception as e:
            print(f"Error controlling {server_config['url']} ({config_file}): {e}")
            success = False
    
    return success

def get_fear_greed_index():
    """Get Fear & Greed index from alternative.me"""
    try:
        response = requests.get('https://api.alternative.me/fng/?limit=48')
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            # Fix timestamp deprecation warning by converting to numeric first
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['value'] = pd.to_numeric(df['value'])
            return df
        else:
            print(f"Error fetching Fear & Greed index: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching Fear & Greed index: {e}")
        return None

def get_cmc_fear_greed():
    """Fetch the Fear & Greed index from CoinMarketCap API"""
    url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical"
    
    headers = {
        'X-CMC_PRO_API_KEY': 'fe42fca1-a0aa-4bb1-9ded-74f52af453bd'
    }
    
    params = {
        'limit': 48  # Get last 48 data points
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if response.status_code == 200 and 'data' in data:
            # Convert the data to DataFrame
            df = pd.DataFrame(data['data'])
            # Fix timestamp deprecation warning by converting to numeric first
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            return df
    except Exception as e:
        print(f"Error fetching CMC fear & greed index: {e}")
    return None

def format_date(x, p):
    """Custom date formatter to show dates at midnight and every 4 hours"""
    dt = mdates.num2date(x)
    if dt.hour == 0:
        return dt.strftime('%Y-%m-%d')
    elif dt.hour % 4 == 0 and dt.minute == 0:
        return f"{dt.hour:02d}:00"
    return ''

def get_btc_trend() -> Tuple[float, pd.DataFrame]:
    """
    Get BTC/USDT price data from FreqTrade's historical data
    Returns:
        Tuple containing (trend_scalar, dataframe with price data)
        trend_scalar is between -1 (strong downtrend) and 1 (strong uptrend)
    """
    try:
        # Use FreqTrade's data directory
        data_dir = Path("user_data/data/binance")
        data_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        btc_file = data_dir / "BTC_USDT-1h.json"  # FreqTrade uses underscore in pair names
        
        # If data file doesn't exist or is too old, download it
        if not btc_file.exists() or (datetime.now() - datetime.fromtimestamp(btc_file.stat().st_mtime)) > timedelta(hours=1):
            print("Downloading fresh BTC data...")
            import subprocess
            subprocess.run([
                "freqtrade", "download-data", 
                "--exchange", "binance",
                "--pairs", "BTC/USDT",
                "--timeframe", "1h",
                "--days", "30",
                "--trading-mode", "spot"  # Ensure we get spot market data
            ], check=True)
        
        # Load the data
        with open(btc_file, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame with named columns
        # FreqTrade format: [timestamp, open, high, low, close, volume]
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # Print column names for debugging
        print("\nAvailable columns:", df.columns.tolist())
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df['close'] = df['close'].astype(float)
        
        # Sort by timestamp to ensure correct order
        df = df.sort_values('timestamp')
        
        # Calculate trend using last 48 hours of data
        df['returns'] = df['close'].pct_change()
        df['smooth_returns'] = df['returns'].rolling(window=24, min_periods=1).mean()  # Use 24h window for trend
        recent_trend = df['smooth_returns'][-48:].mean()
        trend_scalar = np.clip(recent_trend * 100, -1, 1)  # Scale and clip
        
        print(f"\nBTC Trend Analysis:")
        print(f"48h trend: {recent_trend:.2%}")
        print(f"Trend scalar: {trend_scalar:.2f}")
        
        # Print first few rows for debugging
        print("\nFirst few rows of data:")
        print(df[['timestamp', 'close']].head())
        
        return trend_scalar, df
        
    except Exception as e:
        print(f"Error calculating BTC trend: {e}")
        import traceback
        traceback.print_exc()  # Print full error trace for debugging
        return 0.0, pd.DataFrame()

class ConfigListItem(ttk.Frame):
    """Custom widget for config list items with checkbox"""
    def __init__(self, parent, config_path, on_delete=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.config_path = config_path
        self.on_delete = on_delete
        
        # Create checkbox for dry run
        self.dry_run_var = BooleanVar()
        self.dry_run_var.set(self.get_dry_run_status())
        
        # Create container frame for checkbox and label
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.checkbox = ttk.Checkbutton(
            self.content_frame, 
            text="Dry Run",
            variable=self.dry_run_var,
            command=self.on_checkbox_toggle
        )
        self.checkbox.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create label for config path
        self.label = ttk.Label(
            self.content_frame,
            text=str(Path(config_path).name)  # Show only filename
        )
        self.label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Create button frame for reload and delete
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(side=tk.RIGHT)
        
        # Add reload button
        self.reload_btn = ttk.Button(
            self.button_frame,
            text="↻",  # Use reload symbol
            width=3,
            command=self.reload_config
        )
        self.reload_btn.pack(side=tk.LEFT, padx=(0, 2))
        
        # Add delete button
        self.delete_btn = ttk.Button(
            self.button_frame,
            text="×",  # Use × symbol for delete
            width=3,
            command=self.delete_item
        )
        self.delete_btn.pack(side=tk.LEFT)
    
    def get_dry_run_status(self):
        """Read dry run status from config file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return config.get('dry_run', True)  # Default to True if not found
        except Exception as e:
            print(f"Error reading dry run status from {self.config_path}: {e}")
            return True
    
    def on_checkbox_toggle(self):
        """Update dry run setting in config file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            config['dry_run'] = self.dry_run_var.get()
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Updated dry run setting to {self.dry_run_var.get()} in {self.config_path}")
        except Exception as e:
            print(f"Error updating dry run setting in {self.config_path}: {e}")
            # Revert checkbox if update failed
            self.dry_run_var.set(not self.dry_run_var.get())
    
    def delete_item(self):
        """Handle delete button click"""
        if self.on_delete:
            self.on_delete(self)

    def reload_config(self):
        """Reload this config"""
        server_config = get_server_config(self.config_path)
        if not server_config:
            print(f"Invalid config file: {self.config_path}")
            return
        
        try:
            credentials = f"{server_config['username']}:{server_config['password']}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/json'
            }
            
            url = f"{server_config['url']}/api/v1/reload_config"
            response = requests.post(url, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"Successfully reloaded config: {Path(self.config_path).name}")
            else:
                print(f"Failed to reload config: {Path(self.config_path).name} - {response.status_code}")
        except Exception as e:
            print(f"Error reloading config {self.config_path}: {e}")

class ScrollableConfigFrame(ttk.Frame):
    """Scrollable frame for config items"""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack everything
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Store config items
        self.config_items = []
    
    def add_config(self, config_path):
        """Add a new config item to the list"""
        if config_path not in [item.config_path for item in self.config_items]:
            item = ConfigListItem(
                self.scrollable_frame, 
                config_path,
                on_delete=self.remove_item
            )
            item.pack(fill=tk.X, padx=5, pady=2)
            self.config_items.append(item)
    
    def remove_item(self, item):
        """Remove a specific config item"""
        if item in self.config_items:
            item.pack_forget()
            self.config_items.remove(item)
            item.destroy()
    
    def get_all_configs(self):
        """Get list of all config paths"""
        return [item.config_path for item in self.config_items]

class FearGreedViewer:
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Fear & Greed Index Trader")
        self.root.geometry("1400x900")
        
        # Setup config persistence
        self.config_save_file = "fear_greed_configs.json"
        
        # Create main frames
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create left and right control panels
        self.left_controls = ttk.LabelFrame(self.controls_frame, text="Config Management")
        self.left_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.right_controls = ttk.LabelFrame(self.controls_frame, text="Trading Controls")
        self.right_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Setup matplotlib figure with adjusted size and margins
        self.fig = Figure(figsize=(12, 6))
        self.fig.subplots_adjust(right=0.85)  # Adjust right margin to fit legend
        self.ax = self.fig.add_subplot(111)
        
        # Embed matplotlib figure in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()
        
        # Initialize hover annotation
        self.hover_annotation = self.ax.annotate('',
            xy=(0, 0),
            xytext=(20, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
            visible=False)
        
        # Replace config list with new scrollable frame
        self.config_frame = ScrollableConfigFrame(self.left_controls)
        self.config_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Trading controls
        self.ratio_frame = ttk.Frame(self.right_controls)
        self.ratio_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.ratio_frame, text="Manual Ratio:").pack(side=tk.LEFT, padx=5)
        self.ratio_spinbox = ttk.Spinbox(
            self.ratio_frame,
            from_=0.0,
            to=0.5,
            increment=0.01,
            width=10,
            format="%.2f"
        )
        self.ratio_spinbox.set("0.25")  # Default value as string to ensure 2 decimal places
        self.ratio_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.ratio_frame, text="Recommended:").pack(side=tk.LEFT, padx=5)
        self.recommended_label = ttk.Label(self.ratio_frame, text="0.00%")
        self.recommended_label.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        self.control_buttons = ttk.Frame(self.right_controls)
        self.control_buttons.pack(fill=tk.X, pady=5)
        
        # Add browse button first
        self.browse_btn = ttk.Button(
            self.control_buttons, 
            text="+ Add Config", 
            command=self.browse_config,
            style="Add.TButton"
        )
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Then the existing buttons
        self.apply_btn = ttk.Button(self.control_buttons, text="Apply", command=self.apply_ratio)
        self.apply_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(self.control_buttons, text="Stop All", command=self.stop_trading, style="Stop.TButton")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Auto-scale checkbox
        self.auto_scale_var = tk.BooleanVar(value=False)
        self.auto_scale = ttk.Checkbutton(
            self.control_buttons,
            text="Auto Scale",
            variable=self.auto_scale_var,
            command=self.update_auto_modes
        )
        self.auto_scale.pack(side=tk.LEFT, padx=5)
        
        # Create custom style for add button
        style = ttk.Style()
        style.configure("Add.TButton", font=("TkDefaultFont", 10, "bold"))
        style.configure("Stop.TButton", foreground="red")
        
        # Style for stop button
        style = ttk.Style()
        style.configure("Stop.TButton", foreground="red")
        
        # Initialize other variables
        self.auto_scale_enabled = False
        self.current_ratio = None
        self.last_alt_fg_value = None
        self.last_cmc_fg_value = None
        self.last_alt_update_time = None
        self.last_cmc_update_time = None
        self.trading_stopped = False
        
        # Setup timers and data storage
        self.current_df_alt = None
        self.current_df_cmc = None
        self.last_cmc_api_call = None
        self.cmc_update_interval = 3600000  # 1 hour in milliseconds
        self.alt_update_interval = 3600000  # 1 hour in milliseconds
        self.last_btc_update = None
        self.btc_update_interval = 300000  # 5 minutes in milliseconds
        
        # Connect events
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.canvas.mpl_connect('resize_event', self.on_resize)
        
        # Load saved configs
        self.load_saved_configs()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start update loop
        self.update_plot()
        self.schedule_next_update()

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, justify=tk.LEFT,
                            background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            
            widget.tooltip = tooltip
            
        def leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def schedule_next_update(self):
        """Schedule the next plot update"""
        # Use the shortest interval among all data sources
        next_update = min(self.alt_update_interval, self.cmc_update_interval, self.btc_update_interval)
        self.root.after(next_update, self.update_wrapper)
    
    def update_wrapper(self):
        """Wrapper for update_plot that reschedules the next update"""
        self.update_plot()
        self.schedule_next_update()

    def browse_config(self):
        """Open file dialog to browse for config files"""
        file_path = filedialog.askopenfilename(
            title='Select Config File',
            filetypes=[('JSON files', '*.json')],
            initialdir='user_data/configs/'
        )
        
        if file_path:
            self.config_frame.add_config(file_path)

    def apply_ratio(self):
        """Apply the manual ratio to all configs and reload them"""
        try:
            ratio = float(self.ratio_spinbox.get())
            print(f"\nApplying manual ratio {ratio:.2%} to all configs...")
            
            for config_file in self.config_frame.get_all_configs():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    config['tradable_balance_ratio'] = round(ratio, 2)  # Ensure 2 decimal places
                    
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=4)
                    print(f"Updated {Path(config_file).name}")
                    
                except Exception as e:
                    print(f"Error updating {config_file}: {e}")
            
            # Reload all configs
            self.reload_all_configs()
        except ValueError:
            print("Invalid ratio value")

    def calculate_dynamic_ratio(self, fear_greed_value):
        """Calculate recommended ratio based on fear/greed and BTC trend"""
        btc_trend_scalar, _ = get_btc_trend()
        
        # Calculate base ratio from fear/greed
        if fear_greed_value <= 20:
            base_ratio = 0.3 * (1 - fear_greed_value / 20)
        elif fear_greed_value <= 70:
            base_ratio = (fear_greed_value - 20) / (70 - 20) * 0.5
        else:
            base_ratio = 0.5 * (1 - (fear_greed_value - 70) / 30)
        
        # Adjust ratio based on BTC trend
        trend_adjustment = btc_trend_scalar * 0.2
        final_ratio = base_ratio * (1 + trend_adjustment)
        final_ratio = min(0.5, max(0.0, final_ratio))
        
        # Update recommended ratio display
        self.recommended_label.config(text=f"{final_ratio:.2%}")
        
        return final_ratio

    def update_auto_modes(self):
        """Handle auto-scale checkbox changes"""
        self.auto_scale_enabled = self.auto_scale_var.get()
        print(f"Auto Scale set to {self.auto_scale_enabled}")
        
        if self.auto_scale_enabled and not self.trading_stopped:
            # Calculate and apply the recommended ratio immediately when enabled
            if self.current_df_alt is not None:
                latest_alt_value = self.current_df_alt['value'].iloc[-1]
                if self.current_df_cmc is not None:
                    cmc_value = self.current_df_cmc['value'].iloc[-1]
                    combined_value = (latest_alt_value + cmc_value) / 2
                else:
                    combined_value = latest_alt_value
                
                new_ratio = self.calculate_dynamic_ratio(combined_value)
                self.adjust_risk(new_ratio)

    def run(self):
        """Start the main event loop"""
        self.root.mainloop()

    def reload_all_configs(self):
        """Reload all bot configs"""
        success = True
        
        # Get list of config files from the GUI config list
        config_files = list(self.config_frame.get_all_configs())
        if not config_files:
            print("No config files loaded in the list")
            return False
        
        for config_file in config_files:
            server_config = get_server_config(config_file)
            if not server_config:
                print(f"Skipping invalid config file: {config_file}")
                success = False
                continue
            
            try:
                credentials = f"{server_config['username']}:{server_config['password']}"
                encoded_credentials = base64.b64encode(credentials.encode()).decode()
                headers = {
                    'Authorization': f'Basic {encoded_credentials}',
                    'Content-Type': 'application/json'
                }
                
                url = f"{server_config['url']}/api/v1/reload_config"
                response = requests.post(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    print(f"Successfully reloaded config on {server_config['url']} ({config_file})")
                else:
                    print(f"Failed to reload config on {server_config['url']} ({config_file}): {response.status_code}")
                    success = False
            except Exception as e:
                print(f"Error reloading config on {server_config['url']} ({config_file}): {e}")
                success = False
        
        return success

    def update_plot(self):
        now = datetime.now()
        df_alt = get_fear_greed_index()
        
        # Get new CMC data if needed
        if (self.last_cmc_api_call is None or 
            ((now - self.last_cmc_api_call).total_seconds() * 1000 >= self.cmc_update_interval)):
            new_cmc_data = get_cmc_fear_greed()
            if new_cmc_data is not None:
                self.current_df_cmc = new_cmc_data.copy()
                self.last_cmc_api_call = now
                self.last_cmc_update_time = new_cmc_data['timestamp'].iloc[-1]
        
        # Get BTC trend data if needed
        if (self.last_btc_update is None or 
            ((now - self.last_btc_update).total_seconds() * 1000 >= self.btc_update_interval)):
            btc_trend_scalar, btc_df = get_btc_trend()
            self.last_btc_update = now
            self.current_btc_df = btc_df
            self.current_btc_trend = btc_trend_scalar
        else:
            btc_trend_scalar = self.current_btc_trend
            btc_df = self.current_btc_df

        if df_alt is not None:
            # Sort dataframe by timestamp to ensure we get the latest value
            df_alt = df_alt.sort_values('timestamp')
            self.current_df_alt = df_alt
            latest_alt_value = df_alt['value'].iloc[-1]
            latest_alt_time = df_alt['timestamp'].iloc[-1]
            
            # Calculate recommended ratio regardless of auto-scale
            combined_value = latest_alt_value
            if self.current_df_cmc is not None:
                cmc_value = self.current_df_cmc['value'].iloc[-1]
                combined_value = (latest_alt_value + cmc_value) / 2
            
            # Always calculate and update the recommended ratio
            recommended_ratio = self.calculate_dynamic_ratio(combined_value)
            self.recommended_label.config(text=f"{recommended_ratio:.2%}")
            
            # Debug print to check raw values
            print("\nDEBUG Raw values:")
            print(f"Alternative.me last 5 entries:")
            print(df_alt.tail()[['timestamp', 'value']])
            if self.current_df_cmc is not None:
                print(f"\nCMC last 5 entries:")
                print(self.current_df_cmc.tail()[['timestamp', 'value']])
            
            # Check if we have new values
            alt_changed = self.last_alt_fg_value != latest_alt_value
            
            if self.last_alt_fg_value is None or alt_changed:
                self.last_alt_fg_value = latest_alt_value
                self.last_alt_update_time = latest_alt_time

            # Update plot
            self.ax.clear()
            
            # Create second y-axis for BTC trend
            ax2 = self.ax.twinx()
            
            # Add background spans
            self.ax.axhspan(0, 20, alpha=0.1, color='purple', label='Extreme Fear (30-0% trading)')
            self.ax.axhspan(20, 45, alpha=0.1, color='red', label='Fear (0-25% trading)')
            self.ax.axhspan(45, 55, alpha=0.1, color='yellow', label='Neutral (25-35% trading)')
            self.ax.axhspan(55, 70, alpha=0.1, color='lightgreen', label='Greed (35-50% trading)')
            self.ax.axhspan(70, 100, alpha=0.1, color='orange', label='Extreme Greed (50-0% trading)')
            
            # Plot both fear & greed indices
            self.ax.plot(self.current_df_alt['timestamp'], self.current_df_alt['value'], 'b-', 
                        label='Alternative.me Index', linewidth=2, marker='o', markersize=4)
            
            if self.current_df_cmc is not None:
                self.ax.plot(self.current_df_cmc['timestamp'], self.current_df_cmc['value'], 'r--',
                            label='CoinMarketCap Index', linewidth=2, marker='s', markersize=4)
            
            # Plot BTC trend on secondary axis if data available
            if not btc_df.empty:
                # Debug print timestamps
                print("\nTimestamp ranges:")
                print(f"Alt.me: {self.current_df_alt['timestamp'].min()} to {self.current_df_alt['timestamp'].max()}")
                if self.current_df_cmc is not None:
                    print(f"CMC: {self.current_df_cmc['timestamp'].min()} to {self.current_df_cmc['timestamp'].max()}")
                print(f"BTC: {btc_df['timestamp'].min()} to {btc_df['timestamp'].max()}")
                
                # Filter BTC data to match the fear/greed timeframe
                min_time = min(self.current_df_alt['timestamp'].min(), 
                             self.current_df_cmc['timestamp'].min() if self.current_df_cmc is not None else pd.Timestamp.max)
                max_time = max(self.current_df_alt['timestamp'].max(),
                             self.current_df_cmc['timestamp'].max() if self.current_df_cmc is not None else pd.Timestamp.min)
                
                print(f"\nFiltering BTC data between: {min_time} and {max_time}")
                
                # Ensure we have some padding on both ends
                min_time -= pd.Timedelta(hours=1)
                max_time += pd.Timedelta(hours=1)
                
                mask = (btc_df['timestamp'] >= min_time) & (btc_df['timestamp'] <= max_time)
                plot_df = btc_df[mask].copy()
                
                print(f"\nFiltered BTC data points: {len(plot_df)}")
                print("First 5 BTC points:")
                print(plot_df[['timestamp', 'close']].head())
                print("\nLast 5 BTC points:")
                print(plot_df[['timestamp', 'close']].tail())
                
                if not plot_df.empty:
                    # Get all unique timestamps from both indices
                    timestamps = pd.concat([
                        self.current_df_alt['timestamp'],
                        self.current_df_cmc['timestamp'] if self.current_df_cmc is not None else pd.Series()
                    ]).sort_values().unique()
                    
                    # For each timestamp, find the closest BTC price point
                    aligned_data = []
                    for ts in timestamps:
                        # Find closest BTC data point
                        closest_idx = (plot_df['timestamp'] - ts).abs().idxmin()
                        aligned_data.append({
                            'timestamp': ts,
                            'close': plot_df.loc[closest_idx, 'close']
                        })
                    
                    # Create aligned DataFrame
                    aligned_df = pd.DataFrame(aligned_data)
                    
                    # Calculate returns relative to the first price
                    initial_price = aligned_df['close'].iloc[0]
                    aligned_df['cumulative_returns'] = (aligned_df['close'] - initial_price) / initial_price
                    
                    # Apply light smoothing to match Binance's style
                    aligned_df['cumulative_returns'] = aligned_df['cumulative_returns'].rolling(window=2, min_periods=1).mean()
                    
                    print("\nBTC Price Analysis:")
                    print(f"Initial price: ${aligned_df['close'].iloc[0]:,.2f}")
                    print(f"Final price: ${aligned_df['close'].iloc[-1]:,.2f}")
                    print(f"Cumulative returns range:")
                    print(f"Min: {aligned_df['cumulative_returns'].min():.2%}")
                    print(f"Max: {aligned_df['cumulative_returns'].max():.2%}")
                    
                    # Debug print for last few points to check alignment
                    print("\nLast 5 aligned points:")
                    print(aligned_df.tail()[['timestamp', 'close', 'cumulative_returns']])
                    
                    # Plot cumulative returns as percentage
                    ax2.plot(aligned_df['timestamp'], aligned_df['cumulative_returns'] * 100, 'g-', 
                            label='BTC Return', linewidth=1.5)
                    
                    # Set limits and labels for secondary axis with extra padding
                    max_abs_return = max(abs(aligned_df['cumulative_returns'].max()), 
                                       abs(aligned_df['cumulative_returns'].min())) * 100
                    y_limit = max(2, np.ceil(max_abs_return * 1.1))  # Add 10% padding and minimum ±2%
                    ax2.set_ylim(-y_limit, y_limit)
                    ax2.set_ylabel('BTC Return (%)', color='g')
                    ax2.tick_params(axis='y', labelcolor='g')
                    
                    # Add horizontal line at 0 for reference
                    ax2.axhline(y=0, color='g', linestyle=':', alpha=0.3)
                    
                    # Set y-axis ticks to match Binance style, but ensure we cover the full range
                    major_tick = max(2.0, np.ceil(y_limit / 5))  # At least 2% or divide range into ~5 ticks
                    minor_tick = major_tick / 2
                    ax2.yaxis.set_major_locator(plt.MultipleLocator(major_tick))
                    ax2.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick))
                    
                    # Print actual y-axis limits for debugging
                    print(f"\nY-axis limits: {ax2.get_ylim()}")
                    print(f"Major tick interval: {major_tick}%")
                    print(f"Minor tick interval: {minor_tick}%")
            
            # Customize the plot
            self.ax.set_title('Crypto Fear & Greed Indices with BTC Trend (Last 2 Days)', fontsize=14)
            self.ax.set_ylabel('Fear & Greed Index', fontsize=12)
            self.ax.set_xlabel('Time', fontsize=12)
            self.ax.grid(True, alpha=0.3)
            
            # Use legend outside the plot and format x-axis
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, 
                      loc='center left', bbox_to_anchor=(1, 0.5))
            
            self.ax.set_ylim(0, 100)
            self.ax.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
            self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.setp(self.ax.get_xticklabels(), rotation=45)
            
            # Add Last Updated Indicator with color coding
            time_since_update = now - latest_alt_time.replace(tzinfo=None)
            update_color = 'gray' if time_since_update < timedelta(hours=8) else 'red'
            self.ax.text(0.98, 0.02, 
                        f"Alternative.me update: {latest_alt_time.strftime('%Y-%m-%d %H:%M:%S')}", 
                        transform=self.ax.transAxes,
                        ha="right", va="bottom", fontsize=10, color=update_color)
            
            if self.current_df_cmc is not None and self.last_cmc_update_time is not None:
                time_since_cmc = now - self.last_cmc_update_time.replace(tzinfo=None)
                cmc_color = 'gray' if time_since_cmc < timedelta(hours=8) else 'red'
                self.ax.text(0.98, 0.06, 
                           f"CoinMarketCap update: {self.last_cmc_update_time.strftime('%Y-%m-%d %H:%M:%S')}", 
                           transform=self.ax.transAxes,
                           ha="right", va="bottom", fontsize=10, color=cmc_color)
            
            # Add BTC trend info
            if not btc_df.empty:
                self.ax.text(0.98, 0.10,
                           f"BTC Trend Scalar: {btc_trend_scalar:.2f}",
                           transform=self.ax.transAxes,
                           ha="right", va="bottom", fontsize=10, color='green')
            
            # Store DataFrames for hover callback
            self.current_df_alt = df_alt.copy()
            if self.current_df_cmc is not None:
                self.current_df_cmc = self.current_df_cmc.copy()
        
        self.canvas.draw()

    def on_resize(self, event):
        """Handle resize events to ensure hover works after window changes"""
        self.canvas.draw_idle()

    def on_hover(self, event):
        if event.inaxes == self.ax:
            # Find closest point from both datasets
            closest_point = None
            min_distance = float('inf')
            
            if hasattr(self, "current_df_alt") and self.current_df_alt is not None:
                xdata_alt = mdates.date2num(self.current_df_alt['timestamp'])
                ydata_alt = self.current_df_alt['value']
                dist_alt = np.sqrt((xdata_alt - event.xdata)**2 + ((ydata_alt - event.ydata)/100)**2)
                min_dist_alt = np.min(dist_alt)
                if min_dist_alt < min_distance:
                    min_distance = min_dist_alt
                    idx_alt = np.argmin(dist_alt)
                    closest_point = {
                        'time': self.current_df_alt['timestamp'].iloc[idx_alt],
                        'alt_value': self.current_df_alt['value'].iloc[idx_alt],
                        'cmc_value': None
                    }
            
            if hasattr(self, "current_df_cmc") and self.current_df_cmc is not None:
                xdata_cmc = mdates.date2num(self.current_df_cmc['timestamp'])
                ydata_cmc = self.current_df_cmc['value']
                dist_cmc = np.sqrt((xdata_cmc - event.xdata)**2 + ((ydata_cmc - event.ydata)/100)**2)
                min_dist_cmc = np.min(dist_cmc)
                if min_dist_cmc < min_distance:
                    min_distance = min_dist_cmc
                    idx_cmc = np.argmin(dist_cmc)
                    closest_point = {
                        'time': self.current_df_cmc['timestamp'].iloc[idx_cmc],
                        'alt_value': None,
                        'cmc_value': self.current_df_cmc['value'].iloc[idx_cmc]
                    }
            
            if closest_point and min_distance < 0.02:  # Only show tooltip if mouse is close enough
                self.hover_annotation.xy = (mdates.date2num(closest_point['time']), 
                                         closest_point['alt_value'] if closest_point['alt_value'] is not None 
                                         else closest_point['cmc_value'])
                
                text = f"{closest_point['time'].strftime('%Y-%m-%d %H:%M')}"
                if closest_point['alt_value'] is not None:
                    text += f"\nAlternative.me: {closest_point['alt_value']:.1f}"
                if closest_point['cmc_value'] is not None:
                    text += f"\nCoinMarketCap: {closest_point['cmc_value']:.1f}"
                
                # Calculate trading ratio based on available values
                if closest_point['alt_value'] is not None and closest_point['cmc_value'] is not None:
                    combined_value = (closest_point['alt_value'] + closest_point['cmc_value']) / 2
                    ratio = self.calculate_dynamic_ratio(combined_value)
                    text += f"\nCombined Trading: {ratio:.1%}"
                elif closest_point['alt_value'] is not None:
                    ratio = self.calculate_dynamic_ratio(closest_point['alt_value'])
                    text += f"\nTrading: {ratio:.1%}"
                elif closest_point['cmc_value'] is not None:
                    ratio = self.calculate_dynamic_ratio(closest_point['cmc_value'])
                    text += f"\nTrading: {ratio:.1%}"
                
                self.hover_annotation.set_text(text)
                self.hover_annotation.set_visible(True)
                self.canvas.draw_idle()
            else:
                if self.hover_annotation.get_visible():
                    self.hover_annotation.set_visible(False)
                    self.canvas.draw_idle()
        else:
            if self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()

    def stop_trading(self):
        """Stop the bot from opening new trades but keep it running"""
        print("Enabling stop buy mode...")
        
        # Call stopbuy to prevent new trades
        if control_trading(self, 'stopbuy'):
            self.trading_stopped = True
            self.current_ratio = 0.0
            print("Successfully enabled stop buy mode - bot will not enter new trades")
        else:
            print("Failed to enable stop buy mode")

    def adjust_risk(self, new_ratio):
        """Adjust trading risk by updating the ratio"""
        if control_trading(self, 'adjust', balance_ratio=new_ratio):
            self.current_ratio = new_ratio
            print(f"Successfully adjusted ratio to {new_ratio:.1%}")
        else:
            print("Failed to adjust trading ratio")

    def load_saved_configs(self):
        """Load saved config files from JSON"""
        try:
            if Path(self.config_save_file).exists():
                with open(self.config_save_file, 'r') as f:
                    saved_configs = json.load(f)
                    for config in saved_configs:
                        if Path(config).exists():  # Only add if file still exists
                            self.config_frame.add_config(config)
        except Exception as e:
            print(f"Error loading saved configs: {e}")

    def save_configs(self):
        """Save current config files to JSON"""
        try:
            configs = self.config_frame.get_all_configs()
            with open(self.config_save_file, 'w') as f:
                json.dump(configs, f, indent=4)
        except Exception as e:
            print(f"Error saving configs: {e}")

    def on_closing(self):
        """Handle window closing event"""
        self.save_configs()
        self.root.destroy()

def check_freqtrade_connection():
    """Check if FreqTrade is accessible"""
    for server in FREQTRADE_SERVERS:
        try:
            url = f"{server['url']}/api/v1/ping"
            credentials = f"{server['username']}:{server['password']}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/json'
            }
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                print(f"Successfully connected to FreqTrade at {server['url']}")
                return True
            else:
                print(f"FreqTrade at {server['url']} returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"\nERROR: Could not connect to FreqTrade at {server['url']}")
            print("Please ensure:")
            print("1. FreqTrade is running")
            print("2. The API URL and port are correct")
            print("3. API is enabled in your config")
        except Exception as e:
            print(f"Error checking FreqTrade connection: {e}")
    return False

def test_ratio(ratio):
    """Quick test function to set a specific ratio"""
    print(f"\nTesting ratio adjustment to {ratio}")
    if check_freqtrade_connection():
        control_trading(ratio)
    else:
        print("Aborting due to connection issues.")

def update_server_port(port):
    """Update the server port in the configuration"""
    global FREQTRADE_SERVERS
    FREQTRADE_SERVERS[0]["url"] = f"http://127.0.0.1:{port}"
    print(f"Updated server URL to: {FREQTRADE_SERVERS[0]['url']}")

if __name__ == "__main__":
    # Add command line argument handling
    parser = argparse.ArgumentParser(description='Fear & Greed Index Trader')
    parser.add_argument('command', nargs='?', default='view', help='Command to run (test10, test25, test50, test75, test100, check, or view)')
    parser.add_argument('--port', type=int, help='FreqTrade API port (default: 8081)')
    
    args = parser.parse_args()
    
    # Update port if specified
    if args.port:
        update_server_port(args.port)
    
    if args.command == "view":
        viewer = FearGreedViewer()
        viewer.run()
    else:
        # Handle other commands as before
        if args.command == "test10":
            test_ratio(0.1)
        elif args.command == "test25":
            test_ratio(0.25)
        elif args.command == "test50":
            test_ratio(0.5)
        elif args.command == "test75":
            test_ratio(0.75)
        elif args.command == "test100":
            test_ratio(1.0)
        elif args.command == "check":
            check_freqtrade_connection()
        elif args.command.startswith("test"):
            try:
                percentage = float(args.command[4:])
                test_ratio(percentage / 100)
            except ValueError:
                print(f"Invalid test command: {args.command}") 