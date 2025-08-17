# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timezone, timedelta
from freqtrade.persistence import Trade
import logging
import json
from pathlib import Path
from typing import Optional, Union
import os

logger = logging.getLogger(__name__)


class homebrew(IStrategy):
    """
    Strategy using PSAR + price action for entries, RSI for exits with trailing stops
    """
    # Trailing stop settings
    # trailing_stop = True
    # trailing_stop_positive = 0.005  # Start trailing once we have 0.5% profit
    # trailing_stop_positive_offset = 0.01  # Don't trail closer than 1%
    # trailing_only_offset_is_reached = True  # Only trail after offset is reached

    # Basic strategy settings
    minimal_roi = {
        "0": 0.2  # Take profit at 10%
    }
    stoploss = -0.004  # 2% stop loss
    #exit_profit_only = True

    # Strategy parameters
    SMA_WINDOW = 3  # Number of candles for the SMA
    TREND_WINDOW = 10  # Number of candles for general trend SMA
    MIN_SLOPE = 0.2        # Increased from 0.1 to 0.2 based on logs
    MIN_DIVERGENCE = 0.4   # Increased from 0.2 to 0.4 based on logs
    MIN_TREND_SLOPE = 0.05 # General trend threshold


    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Create trades directory if it doesn't exist
        self.trades_log_path = Path('user_data/trades_log_8hr')
        self.trades_log_path.mkdir(parents=True, exist_ok=True)
        self.current_trades = {}
        self.window_in_candles = None
        
        logger.warning(f"Strategy initialized - logs will be saved to: {self.trades_log_path.absolute()}")
        
        # Check if directory exists and is writable
        if not self.trades_log_path.exists():
            logger.error(f"Trade logs directory does not exist: {self.trades_log_path.absolute()}")
        elif not os.access(self.trades_log_path, os.W_OK):
            logger.error(f"Trade logs directory is not writable: {self.trades_log_path.absolute()}")
        else:
            logger.warning(f"Trade logs directory is ready for writing")

        # List any existing trade log files
        existing_logs = list(self.trades_log_path.glob('trades_*.json'))
        logger.warning(f"Found {len(existing_logs)} existing trade log files: {existing_logs}")

    def log_trade_details(self, pair: str, action: str, data: dict) -> None:
        """Log trade details to JSON file"""
        filename = self.trades_log_path / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
        logger.warning(f"Attempting to log {action} for {pair} to {filename}")
        
        try:
            if action == 'entry':
                logger.warning(f"Recording entry for {pair}")
                self.current_trades[pair] = {
                    'pair': pair,
                    'entry_time': str(datetime.now()),
                    'entry_reason': {
                        'price': float(data['ha_typical']),
                        'psar': float(data['sar']),
                        'sma': float(data['typical_sma']),
                        'sma_slope': float(data['sma_slope']),
                        'ha_slope': float(data['ha_slope']),
                        'avg_slope': float(data['avg_slope']),
                        'divergence': float(data['divergence'])
                    }
                }
                logger.warning(f"Entry recorded for {pair} in memory")
                
                # Write to file even for entries
                trades = []
                if filename.exists():
                    try:
                        with open(filename, 'r') as f:
                            trades = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not read existing trades file {filename}, starting fresh")
                        trades = []
                
                # Add the entry as a pending trade
                trades.append(self.current_trades[pair])
                
                with open(filename, 'w') as f:
                    json.dump(trades, f, indent=4)
                logger.warning(f"Entry saved to file for {pair}")
                
            elif action == 'exit':
                logger.warning(f"Processing exit for {pair}")
                if pair in self.current_trades:
                    logger.warning(f"Found existing trade for {pair}, recording exit")
                    self.current_trades[pair]['exit_time'] = str(datetime.now())
                    self.current_trades[pair]['exit_reason'] = {
                        'price': float(data['ha_typical']),
                        'psar': float(data['sar']),
                        'typical_sma': float(data['typical_sma']),
                        'sma_slope': float(data['sma_slope']),
                        'ha_slope': float(data['ha_slope']),
                        'divergence': float(data['divergence']),
                        'exit_reason': data['exit_reason'],
                        'psar_cross': data['psar_cross'],
                        'red_candle': data['red_candle']
                    }
                    
                    # Read existing trades
                    trades = []
                    if filename.exists():
                        try:
                            with open(filename, 'r') as f:
                                trades = json.load(f)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not read existing trades file {filename}, starting fresh")
                            trades = []
                    
                    # Update or append the trade
                    updated = False
                    for i, trade in enumerate(trades):
                        if trade.get('pair') == pair and 'exit_time' not in trade:
                            trades[i] = self.current_trades[pair]
                            updated = True
                            break
                    
                    if not updated:
                        trades.append(self.current_trades[pair])
                    
                    # Write back to file
                    with open(filename, 'w') as f:
                        json.dump(trades, f, indent=4)
                    
                    logger.warning(f"Successfully saved trade log for {pair}")
                    del self.current_trades[pair]
                else:
                    logger.warning(f"No entry found for {pair} when trying to log exit")
                    
        except Exception as e:
            logger.error(f"Error in log_trade_details: {str(e)}")
            logger.exception(e)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # PSAR
        dataframe['sar'] = ta.SAR(dataframe)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Calculate Heikin Ashi
        dataframe = self.heikin_ashi(dataframe)
        dataframe['ha_typical'] = (dataframe['ha_high'] + dataframe['ha_low'] + dataframe['ha_close']) / 3
        
        # Add SMA of typical price
        dataframe['typical_sma'] = dataframe['ha_typical'].rolling(self.SMA_WINDOW).mean()

        # Add general trend SMA
        dataframe['trend_sma'] = dataframe['ha_typical'].rolling(self.TREND_WINDOW).mean()
        
        # Calculate trend slope
        dataframe['trend_slope'] = (
            (dataframe['trend_sma'] - dataframe['trend_sma'].shift(3)) /
            dataframe['trend_sma'] * 100  # Convert to percentage change
        )

        # Calculate slopes over 3 candles for smoother measurement
        dataframe['sma_slope'] = (
            (dataframe['typical_sma'] - dataframe['typical_sma'].shift(3)) /
            dataframe['typical_sma'] * 100  # Convert to percentage change
        )
        dataframe['ha_slope'] = (
            (dataframe['ha_typical'] - dataframe['ha_typical'].shift(3)) /
            dataframe['ha_typical'] * 100  # Convert to percentage change
        )
            # SMA below HA typical = room to move up
        sma_below_price = dataframe['typical_sma'] < dataframe['ha_typical']

        # SMA below price cross
        dataframe['sma_below_price_cross'] = (
            (dataframe['typical_sma'] < dataframe['ha_typical']) &
            (dataframe['typical_sma'].shift(1) > dataframe['ha_typical'].shift(1))
        )
        
        # Calculate average slope
        dataframe['avg_slope'] = (dataframe['sma_slope'] + dataframe['ha_slope']) / 2
        
        # Divergence: positive when HA is moving faster than SMA
        dataframe['divergence'] = dataframe['ha_slope'] - dataframe['sma_slope']
        
        return dataframe

    def heikin_ashi(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate Heikin Ashi values
        """
        df = dataframe.copy()
        
        # Get HA close
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Get HA open
        df['ha_open'] = 0.0
        for i in range(len(df)):
            if i == 0:
                df['ha_open'].iat[0] = (df['open'].iat[0] + df['close'].iat[0]) / 2
            else:
                df['ha_open'].iat[i] = (df['ha_open'].iat[i-1] + df['ha_close'].iat[i-1]) / 2
        
        # Get HA high and low
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            # PSAR cross: current PSAR below price, but previous PSAR was above
            psar_below = dataframe['sar'] < dataframe['ha_typical']
            psar_cross = (
                psar_below &
                (dataframe['sar'].shift(1) > dataframe['ha_typical'].shift(1))
            )


            
            # green candle
            green_candle = (
                (dataframe['ha_close'] > dataframe['ha_open'])
            )
            
            # SMA below HA typical
            sma_below_price = (
                (dataframe['typical_sma'] < dataframe['ha_typical'])
            )

            #sma below price cross in the last 2 candles
            sma_below_price_cross = (
                dataframe['sma_below_price_cross'].shift(1) |
                dataframe['sma_below_price_cross'].shift(2) 
            
            )
            
            # Both lines moving up at minimum rate
            steep_enough = (
                (dataframe['sma_slope'] > self.MIN_SLOPE) & 
                (dataframe['ha_slope'] > self.MIN_SLOPE * 2)  # HA should be moving twice as fast
            )
            
            # HA moving up faster than SMA
            diverging_enough = dataframe['divergence'] > self.MIN_DIVERGENCE
            
            # General trend is up
            trend_up = dataframe['trend_slope'] > self.MIN_TREND_SLOPE

            # Debug logging with colors for the last candle
            if not dataframe.empty:
                last = dataframe.iloc[-1]
                
                # ANSI color codes
                GREEN = "\033[32m"     # Green for passing conditions
                RED = "\033[31m"       # Red for failing conditions
                YELLOW = "\033[33m"    # Yellow for labels
                RESET = "\033[0m"      # Reset color
                
                logger.warning(
                    f"{YELLOW}Entry {metadata['pair']}: "
                    f"SMA:{GREEN if last['sma_slope'] > self.MIN_SLOPE else RED}{last['sma_slope']:.2f}%{RESET} "
                    f"HA:{GREEN if last['ha_slope'] > self.MIN_SLOPE * 2 else RED}{last['ha_slope']:.2f}%{RESET} "
                    f"Div:{GREEN if last['divergence'] > self.MIN_DIVERGENCE else RED}{last['divergence']:.3f}{RESET} "
                    f"Trend:{GREEN if last['trend_slope'] > self.MIN_TREND_SLOPE else RED}{last['trend_slope']:.2f}%{RESET} "
                    f"Green:{GREEN if green_candle.iloc[-1] else RED}{'✓' if green_candle.iloc[-1] else '✗'}{RESET} "
                    f"SMA<Price:{GREEN if sma_below_price.iloc[-1] else RED}{'✓' if sma_below_price.iloc[-1] else '✗'}{RESET}"
                )
            
            entry_conditions = (
                #psar_cross &      
                green_candle &    
                sma_below_price & 
                steep_enough &    
                #diverging_enough &
                trend_up &
                sma_below_price_cross         
            )
            
            # Set enter_long flag first
            dataframe.loc[entry_conditions, 'enter_long'] = 1
            
            # Only log trades that we actually entered
            entry_rows = dataframe[entry_conditions & (dataframe['enter_long'] == 1)]
            if not entry_rows.empty:
                # Log only the first entry signal in this candle set
                entry_idx = entry_rows.index[0]
                self.log_trade_details(metadata['pair'], 'entry', {
                    'ha_typical': float(dataframe.loc[entry_idx, 'ha_typical']),
                    'sar': float(dataframe.loc[entry_idx, 'sar']),
                    'typical_sma': float(dataframe.loc[entry_idx, 'typical_sma']),
                    'sma_slope': float(dataframe.loc[entry_idx, 'sma_slope']),
                    'ha_slope': float(dataframe.loc[entry_idx, 'ha_slope']),
                    'divergence': float(dataframe.loc[entry_idx, 'divergence']),
                    'avg_slope': float(dataframe.loc[entry_idx, 'avg_slope'])
                })

        except Exception as e:
            logger.error(f"Entry error for {metadata['pair']}: {str(e)}")
        
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # ANSI color codes
        GREEN = "\033[32m"     # Green for passing conditions
        RED = "\033[31m"       # Red for failing conditions
        YELLOW = "\033[33m"    # Yellow for labels
        RESET = "\033[0m"      # Reset color
        
        # Check all exit conditions
        psar_cross = (last_candle['sar'] > last_candle['ha_typical'] and 
                     dataframe['sar'].iloc[-2] < dataframe['ha_typical'].iloc[-2])
        
        red_candle = (last_candle['ha_close'] < last_candle['ha_open'])
        red_candle_size = ((last_candle['ha_open'] - last_candle['ha_close']) / last_candle['ha_open']) if red_candle else 0
        significant_red = red_candle and red_candle_size > 0.001
        
        MIN_SLOPE = 0.2
        scale_factor = last_candle['sma_slope'] / MIN_SLOPE
        required_convergence = -0.4 * (1 + scale_factor)
        strong_convergence = (last_candle['sma_slope'] < MIN_SLOPE and 
                            last_candle['divergence'] < required_convergence)
        
        slope_reversal = last_candle['ha_slope'] < -0.0
        
        # Debug logging with colors
        logger.warning(
            f"{YELLOW}Exit {pair}: "
            f"PSAR:{GREEN if psar_cross else RED}{'✓' if psar_cross else '✗'}{RESET} "
            f"Red:{GREEN if significant_red else RED}{'✓' if significant_red else '✗'}{RESET}({red_candle_size:.3f}%) "
            f"Conv:{GREEN if strong_convergence else RED}{'✓' if strong_convergence else '✗'}{RESET}(SMA:{last_candle['sma_slope']:.2f}% Div:{last_candle['divergence']:.3f}) "
            f"Slope:{GREEN if slope_reversal else RED}{'✓' if slope_reversal else '✗'}{RESET}({last_candle['ha_slope']:.2f}%)"
        )
        
        # Exit on PSAR cross above
        if psar_cross:
            self.log_trade_details(pair, 'exit', {
                'ha_typical': last_candle['ha_typical'],
                'sar': last_candle['sar'],
                'typical_sma': last_candle['typical_sma'],
                'sma_slope': last_candle['sma_slope'],
                'ha_slope': last_candle['ha_slope'],
                'divergence': last_candle['divergence'],
                'exit_reason': 'psar_cross_exit',
                'psar_cross': True,
                'red_candle': False
            })
            return "psar_cross_exit"
        
        # Exit on first significant red candle
        if significant_red:
            self.log_trade_details(pair, 'exit', {
                'ha_typical': last_candle['ha_typical'],
                'sar': last_candle['sar'],
                'typical_sma': last_candle['typical_sma'],
                'sma_slope': last_candle['sma_slope'],
                'ha_slope': last_candle['ha_slope'],
                'divergence': last_candle['divergence'],
                'exit_reason': 'red_candle_exit',
                'psar_cross': False,
                'red_candle': True
            })
            return "red_candle_exit"
            
        # Exit on strong convergence 
        if strong_convergence:
            self.log_trade_details(pair, 'exit', {
                'ha_typical': last_candle['ha_typical'],
                'sar': last_candle['sar'],
                'typical_sma': last_candle['typical_sma'],
                'sma_slope': last_candle['sma_slope'],
                'ha_slope': last_candle['ha_slope'],
                'divergence': last_candle['divergence'],
                'required_convergence': required_convergence,
                'exit_reason': 'strong_convergence_exit',
                'psar_cross': False,
                'red_candle': False
            })
            return "strong_convergence_exit"

        # Separate check for slope reversal
        if slope_reversal:
            self.log_trade_details(pair, 'exit', {
                'ha_typical': last_candle['ha_typical'],
                'sar': last_candle['sar'],
                'typical_sma': last_candle['typical_sma'],
                'sma_slope': last_candle['sma_slope'],
                'ha_slope': last_candle['ha_slope'],
                'divergence': last_candle['divergence'],
                'exit_reason': 'slope_reversal_exit',
                'psar_cross': False,
                'red_candle': False
            })
            return "slope_reversal_exit"
            
        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Mirror custom_exit conditions here for plotting purposes.
        The actual exits are still handled by custom_exit.
        """
        # PSAR cross exit
        psar_cross = (
            (dataframe['sar'] > dataframe['ha_typical']) &
            (dataframe['sar'].shift(1) < dataframe['ha_typical'].shift(1))
        )
        
        # Red candle exit
        red_candle = (dataframe['ha_close'] < dataframe['ha_open'])
        red_candle_size = ((dataframe['ha_open'] - dataframe['ha_close']) / dataframe['ha_open'])
        significant_red = red_candle & (red_candle_size > 0.001)
        
        # Strong convergence exit
        MIN_SLOPE = 0.2
        scale_factor = dataframe['sma_slope'] / MIN_SLOPE
        required_convergence = -0.4 * (1 + scale_factor)
        strong_convergence = (
            (dataframe['sma_slope'] < MIN_SLOPE) &
            (dataframe['divergence'] < required_convergence)
        )
        
        # Slope reversal exit
        slope_reversal = dataframe['ha_slope'] < -0.0
        
        # # Combine all exit conditions
        # dataframe.loc[
        #     psar_cross |
        #     significant_red |
        #     strong_convergence |
        #     slope_reversal,
        #     'exit_long'
        # ] = 1

        return dataframe

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Configure connection pooling for many concurrent trades
        """
        try:
            if hasattr(self, 'exchange'):
                # Configure the connection pool for high concurrency
                adapter = self.exchange._session.adapters['https://']
                adapter.poolmanager.connection_pool_kw.update({
                    'maxsize': 100,          # Increase max connections
                    'retries': 3,            # Number of retries
                    'pool_block': False      # Don't block when pool is full
                })
                
                # Log current settings
                pool_size = adapter.poolmanager.connection_pool_kw['maxsize']
                logger.info(f"Connection Pool Size: {pool_size}")
        except Exception as e:
            logger.warning(f"Could not configure connection pool: {e}")

    def plot_config(self):
        return {
            'main_plot': {
                'sar': {'color': 'magenta'},
                'ha_typical': {'color': 'white'},
                'typical_sma': {'color': 'yellow'},
                'price_position': {'color': 'orange'},
            },
            'subplots': {
                "Slopes": {
                    'sma_slope': {'color': 'yellow'},
                    'ha_slope': {'color': 'white'},
                    'avg_slope': {'color': 'green'},
                },
                "Divergence": {
                    'divergence': {'color': 'purple'},
                }
            }
        }
