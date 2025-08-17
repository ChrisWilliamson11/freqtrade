import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, CategoricalParameter
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy.interface import IStrategy

# Add the strategy directory to the Python path
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)

class NFI5MOHO_Dynamic(IStrategy):
    """
    Dynamic version of NFI5MOHO strategy that adjusts parameters based on market conditions
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.08,
        "10": 0.04,
        "30": 0.02,
        "60": 0.01
    }

    # Stoploss
    stoploss = -0.10

    # Buy hyperopt parameters
    buy_params = {
    'buy_chop_min_19': 42.1,
    'buy_rsi_1h_min_19': 62.6,
}

    # Sell hyperopt parameters
    sell_params = {
    'sell_rsi_bb_1': 69.4,
    'sell_rsi_bb_2': 73.8,
    'sell_rsi_main_3': 80.0,
}

    # Dynamic Parameters based on market conditions
    market_condition = None
    last_optimization_time = None
    optimization_interval = timedelta(days=1)  # Optimize daily

    # Buy hyperopt parameters
    buy_rsi_1h_min_19 = DecimalParameter(20.0, 70.0, decimals=1, default=60.0, space='buy', optimize=True)
    buy_chop_min_19 = DecimalParameter(20.0, 60.0, decimals=1, default=45.0, space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, decimals=1, default=3.0, space='buy', optimize=True)
    ewo_low = DecimalParameter(-20.0, -8.0, decimals=1, default=-8.0, space='buy', optimize=True)
    low_offset_ema = DecimalParameter(0.9, 0.99, decimals=3, default=0.965, space='buy', optimize=True)
    low_offset_sma = DecimalParameter(0.9, 0.99, decimals=3, default=0.955, space='buy', optimize=True)

    # Sell hyperopt parameters
    high_offset_ema = DecimalParameter(0.99, 1.1, decimals=3, default=1.047, space='sell', optimize=True)
    high_offset_sma = DecimalParameter(0.99, 1.1, decimals=3, default=1.051, space='sell', optimize=True)
    sell_rsi_bb_1 = DecimalParameter(60.0, 80.0, decimals=1, default=79.5, space='sell', optimize=True)
    sell_rsi_bb_2 = DecimalParameter(72.0, 90.0, decimals=1, default=81, space='sell', optimize=True)
    sell_rsi_main_3 = DecimalParameter(77.0, 90.0, decimals=1, default=82, space='sell', optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.market_condition = None
        self.last_optimization_time = None

    def analyze_market_condition(self, dataframe: DataFrame) -> str:
        """
        Analyze current market condition using various indicators
        Returns: 'trending', 'ranging', or 'volatile'
        """
        # Calculate market condition indicators
        volatility = dataframe['close'].pct_change().std()
        adx = ta.ADX(dataframe)
        atr = ta.ATR(dataframe)
        
        # Get the latest values
        current_adx = adx.iloc[-1]
        current_volatility = volatility
        current_atr = atr.iloc[-1]
        
        # Determine market condition
        if current_adx > 25:  # Strong trend
            if current_volatility > 0.02:  # High volatility
                return 'volatile'
            return 'trending'
        else:
            if current_volatility > 0.02:  # High volatility
                return 'volatile'
            return 'ranging'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds various indicators to the dataframe
        """
        # EWO - Elliot Wave Oscillator
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ewo'] = (dataframe['ema_8'] - dataframe['ema_200']) / dataframe['ema_200'] * 100

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_1h'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # CHOP
        window = 14
        atr1 = ta.ATR(dataframe, timeperiod=1)
        dataframe['chop'] = 100 * atr1 / (ta.MAX(dataframe['high'], window) - ta.MIN(dataframe['low'], window))

        # Check if we need to optimize
        current_time = datetime.now()
        if (self.last_optimization_time is None or 
            current_time - self.last_optimization_time >= self.optimization_interval):
            
            # Analyze market condition
            new_condition = self.analyze_market_condition(dataframe)
            
            # If market condition changed, adjust parameters
            if new_condition != self.market_condition:
                self.market_condition = new_condition
                logger.info(f"Market condition changed to: {new_condition}")
                
                # Update parameter values based on market condition
                if new_condition == 'trending':
                    self.buy_rsi_1h_min_19.value = 50.0
                    self.buy_chop_min_19.value = 25.0
                    self.ewo_high.value = 4.0
                    self.ewo_low.value = -10.0
                    self.low_offset_ema.value = 0.984
                    self.low_offset_sma.value = 0.975
                elif new_condition == 'ranging':
                    self.buy_rsi_1h_min_19.value = 60.0
                    self.buy_chop_min_19.value = 45.0
                    self.ewo_high.value = 3.0
                    self.ewo_low.value = -8.0
                    self.low_offset_ema.value = 0.965
                    self.low_offset_sma.value = 0.955
                else:  # volatile
                    self.buy_rsi_1h_min_19.value = 70.0
                    self.buy_chop_min_19.value = 60.0
                    self.ewo_high.value = 2.5
                    self.ewo_low.value = -6.0
                    self.low_offset_ema.value = 0.945
                    self.low_offset_sma.value = 0.935
            
            self.last_optimization_time = current_time

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy strategy
        """
        dataframe.loc[
            (
                (dataframe['ewo'] > self.ewo_high.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_19.value) &
                (dataframe['chop'] > self.buy_chop_min_19.value)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sell strategy
        """
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi_bb_1.value) &
                (dataframe['close'] > dataframe['bb_upperband'] * self.high_offset_ema.value)
            ),
            'sell'] = 1

        return dataframe 