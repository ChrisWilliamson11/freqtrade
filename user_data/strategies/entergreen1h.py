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


class entergreen1h(IStrategy):
    """
    Simple strategy using Heikin Ashi candles:
    - Enter on green candle
    - Exit on red candle
    """
    # ROI table:
    minimal_roi = {
        "0": 0.08,
        "10": 0.04,
        "30": 0.02,
        "60": 0.01
    }
    stoploss = -0.02  # 2% stop loss
    
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate Heikin Ashi
        dataframe = self.heikin_ashi(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter on green Heikin Ashi candle
        dataframe.loc[
            (dataframe['ha_close'] > dataframe['ha_open']),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit on red Heikin Ashi candle
        dataframe.loc[
            (dataframe['ha_close'] < dataframe['ha_open']),
            'exit_long'
        ] = 1
        return dataframe

    def __init__(self, config: dict) -> None:
        super().__init__(config)

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


