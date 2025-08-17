from typing import Dict
import logging
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange.exchange_utils import *
from freqtrade.strategy import IStrategy, RealParameter
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class ExampleLSTMStrategy(IStrategy):
    """
    This is an example strategy that uses the LSTMRegressor model to predict the target score.
    Use at your own risk.
    This is a simple example strategy and should be used for educational purposes only.
    """
    # Hyperspace parameters:
    buy_params = {
        "threshold_buy": 0.05,
        "w0": 0.54347,
        "w1": 0.82226,
        "w2": 0.56675,
        "w3": 0.77918,
        "w4": 0.98488,
        "w5": 0.31368,
        "w6": 0.75916,
        "w7": 0.09226,
        "w8": 0.85667,
    }

    sell_params = {
        "threshold_sell": -0.05,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.05,      # 5% profit is good enough to exit
        "30": 0.025,    # After 30 minutes, exit at 2.5% profit
        "60": 0.01,     # After 60 minutes, exit at 1% profit
        "120": 0        # After 2 hours, exit at any profit
    }

    # Stoploss:
    stoploss = -0.15    # Changed from -0.2 to limit losses

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005          # Start trailing at 0.5% profit
    trailing_stop_positive_offset = 0.01    # Don't trail until 1% profit
    trailing_only_offset_is_reached = True
    timeframe = "5m"
    can_short = False
    use_exit_signal = True
    process_only_new_candles = True

    startup_candle_count = 20

    threshold_buy = RealParameter(-1, 1, default=0, space='buy')
    threshold_sell = RealParameter(-1, 1, default=0, space='sell')

    # Weights for calculating the aggregate score - the sum of all weighted normalized indicators has to be 1!
    w0 = RealParameter(0, 1, default=0.10, space='buy')
    w1 = RealParameter(0, 1, default=0.15, space='buy')
    w2 = RealParameter(0, 1, default=0.10, space='buy')
    w3 = RealParameter(0, 1, default=0.15, space='buy')
    w4 = RealParameter(0, 1, default=0.10, space='buy')
    w5 = RealParameter(0, 1, default=0.10, space='buy')
    w6 = RealParameter(0, 1, default=0.10, space='buy')
    w7 = RealParameter(0, 1, default=0.05, space='buy')
    w8 = RealParameter(0, 1, default=0.15, space='buy')

    # Add this near the top of the class
    plot_config = {
        'main_plot': {
            'bb_upperband': {'color': 'grey'},
            'bb_middleband': {'color': 'red'},
            'bb_lowerband': {'color': 'grey'},
        },
        'subplots': {
            "Predictions 5m": {
                '&-target_mean_5m': {'color': 'blue'},
            },
            "Predictions 1h": {
                '&-target_mean_1h': {'color': 'green'},
            },
            "Predictions 8h": {
                '&-target_mean_8h': {'color': 'red'},
            },
            "Signals": {
                'do_predict': {'color': 'brown'},
            }
        }
    }

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):
        """
        *Only keep absolutely essential features to minimize potential data issues*
        """
        # Basic price features with consistent window size
        window = period
        
        # Price and volume changes
        dataframe["%-close"] = dataframe["close"].pct_change(periods=1)
        dataframe["%-volume"] = dataframe["volume"].pct_change(periods=1)
        
        # Technical indicators
        dataframe["%-sma"] = ta.SMA(dataframe["close"], timeperiod=window)
        dataframe["%-rsi"] = ta.RSI(dataframe["close"], timeperiod=window)
        
        # Rolling statistics
        dataframe["%-close_mean"] = dataframe["close"].rolling(window=window, min_periods=1).mean()
        dataframe["%-close_std"] = dataframe["close"].rolling(window=window, min_periods=1).std()
        
        # Clean all features
        for col in dataframe.columns:
            if col.startswith("%-"):
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
                dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0)
                dataframe[col] = dataframe[col].ffill().bfill()
        
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        # Basic price changes
        dataframe["%-price_change"] = dataframe["close"].pct_change()
        dataframe["%-volume_change"] = dataframe["volume"].pct_change()
        
        # Fill NaN values with 0 directly
        dataframe = dataframe.fillna(0)  # This usage is still supported
        
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        # Convert to datetime if not already
        if not isinstance(dataframe['date'].iloc[0], pd.Timestamp):
            dataframe['date'] = pd.to_datetime(dataframe['date'])
        
        # Add basic time features
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour"] = dataframe["date"].dt.hour
        
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """Calculate targets for each timeframe"""
        
        window = 5  # Use same window size as indicator_periods_candles
        
        # Calculate targets for each timeframe
        for tf in ['5m', '1h', '8h']:
            # Base target
            dataframe[f'&-target_{tf}'] = dataframe['close'].pct_change(periods=1)
            
            # Rolling statistics
            dataframe[f'&-target_mean_{tf}'] = dataframe[f'&-target_{tf}'].rolling(
                window=window, min_periods=1).mean()
            dataframe[f'&-target_std_{tf}'] = dataframe[f'&-target_{tf}'].rolling(
                window=window, min_periods=1).std()
            
            # Clean up NaN/inf values
            for col in [f'&-target_{tf}', f'&-target_mean_{tf}', f'&-target_std_{tf}']:
                dataframe[col] = dataframe[col].replace([np.inf, -np.inf], 0)
                dataframe[col] = dataframe[col].fillna(0)
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # First calculate our base indicators
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['ma'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['bb_upperband'], dataframe['bb_middleband'], dataframe['bb_lowerband'] = ta.BBANDS(dataframe['close'])
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['atr'] = ta.ATR(dataframe)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['roc'] = ta.ROC(dataframe)
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['macdsignal'] = ta.MACD(dataframe)['macdsignal']
        dataframe['macdhist'] = ta.MACD(dataframe)['macdhist']
        dataframe['momentum'] = ta.MOM(dataframe)

        # Then initialize FreqAI
        self.freqai_info = self.config["freqai"]
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on FreqAI predictions from multiple timeframes
        """
        df["enter_long"] = 0
        df["enter_tag"] = ""
        
        # Entry signal when predictions from all timeframes are above threshold
        df.loc[
            (df['do_predict'] == 1) &  # FreqAI is ready to make predictions
            (df['&-target_mean_5m'].astype(float) > self.threshold_buy.value) &  # 5m prediction
            (df['&-target_mean_1h'].astype(float) > self.threshold_buy.value) &  # 1h prediction
            (df['&-target_mean_8h'].astype(float) > self.threshold_buy.value) &  # 8h prediction
            (df['volume'] > 0),  # Ensure volume exists
            'enter_long'
        ] = 1
        
        df.loc[df['enter_long'] == 1, 'enter_tag'] = 'lstm_multi_tf_buy'
        
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on FreqAI predictions from multiple timeframes
        """
        df["exit_long"] = 0
        df["exit_tag"] = ""
        
        # Exit signal when predictions from any timeframe are below threshold
        df.loc[
            (df['do_predict'] == 1) &  # FreqAI is ready to make predictions
            (
                (df['&-target_mean_5m'].astype(float) < self.threshold_sell.value) |  # 5m prediction
                (df['&-target_mean_1h'].astype(float) < self.threshold_sell.value) |  # 1h prediction
                (df['&-target_mean_8h'].astype(float) < self.threshold_sell.value)    # 8h prediction
            ),
            'exit_long'
        ] = 1
        
        df.loc[df['exit_long'] == 1, 'exit_tag'] = 'lstm_multi_tf_sell'
        
        return df
