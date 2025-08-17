# for live trailing_stop = False and use_custom_stoploss = True
# for backtest trailing_stop = True and use_custom_stoploss = False

# --- Do not remove these libs ---
# --- Do not remove these libs ---
from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import pandas as pd

# @Rallipanos
# @pluxury

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 12,
    "ewo_high": 4.0,
    "ewo_high_2": -2.0,
    "ewo_low": -10.0,
    "lookback_candles": 24,
    "low_offset": 0.975,
    "low_offset_2": 0.955,
    "profit_threshold": 1.015,
    "rsi_buy": 65
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 16,
    "high_offset": 1.084,
    "high_offset_2": 1.401,
    "pHSL": -0.15,
    "pPF_1": 0.016,
    "pPF_2": 0.024,
    "pSL_1": 0.014,
    "pSL_2": 0.022
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


def HA(dataframe, smooth=None):
    df = dataframe.copy()
    
    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close'])/4

    # Calculate HA Open
    df['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1))/2
    
    # Set first HA-Open value using the first row's open price
    df.loc[df.index[0], 'HA_Open'] = df['open'].iloc[0]

    # Calculate HA High and Low
    df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)

    if smooth is not None:
        df['HA_Close'] = ta.EMA(df['HA_Close'], smooth)
        df['HA_Open'] = ta.EMA(df['HA_Open'], smooth)
        df['HA_High'] = ta.EMA(df['HA_High'], smooth)
        df['HA_Low'] = ta.EMA(df['HA_Low'], smooth)

    return df


class NASOS8h(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.3,      # 20% immediate profit
        "12": 0.10,     # 4 days (12 * 8h = 96h = 4 days)
        "24": 0.05,     # 8 days
        "48": 0.025     # 16 days
    }

    # Stoploss:
    stoploss = -0.20

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=False)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=False)
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(
        1, 24, default=buy_params['lookback_candles'], space='buy', optimize=True)

    profit_threshold = DecimalParameter(1.0, 1.03,
                                        default=buy_params['profit_threshold'], space='buy', optimize=True)

    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=False)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=False)

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=False)

    rsi_buy = IntParameter(50, 100, default=buy_params['rsi_buy'], space='buy', optimize=False)

    # trailing stoploss hyperopt parameters
    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.15, decimals=3,
                            space='sell', optimize=False, load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3,
                             space='sell', optimize=False, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.014, decimals=3,
                             space='sell', optimize=False, load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.024, decimals=3,
                             space='sell', optimize=False, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.022, decimals=3,
                             space='sell', optimize=False, load=True)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '8h'
    inf_1h = '8h'

    process_only_new_candles = True
    startup_candle_count = 100
    use_custom_stoploss = False

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    # Custom Trailing Stoploss by Perkmeister

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1)*(SL_2 - SL_1)/(PF_2 - PF_1))
        else:
            sl_profit = HSL

        # if current_profit < 0.001 and current_time - timedelta(minutes=600) > trade.open_date_utc:
        #     return -0.005

        return stoploss_from_open(sl_profit, current_profit)

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):  # *1.2
                    return False

        # slippage
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '8h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        
        if informative_1h.empty:
            return informative_1h

        # Calculate Heikin Ashi for informative timeframe
        ha = HA(informative_1h, smooth=None)
        informative_1h['ha_open'] = ha['HA_Open']
        informative_1h['ha_high'] = ha['HA_High']
        informative_1h['ha_low'] = ha['HA_Low']
        informative_1h['ha_close'] = ha['HA_Close']
        
        # Calculate indicators on the higher timeframe data
        informative_1h['ema_50'] = ta.EMA(informative_1h['ha_close'], timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h['ha_close'], timeperiod=200)
        informative_1h['rsi'] = ta.RSI(informative_1h['ha_close'], timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband'] = bollinger['lower']
        informative_1h['bb_middleband'] = bollinger['mid']
        informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # First calculate Heikin Ashi
        ha = HA(dataframe, smooth=None)
        
        # Use HA values for all calculations
        dataframe['ha_open'] = ha['HA_Open']
        dataframe['ha_high'] = ha['HA_High']
        dataframe['ha_low'] = ha['HA_Low']
        dataframe['ha_close'] = ha['HA_Close']

        # Calculate all ma_buy values using HA close
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe['ha_close'], timeperiod=val)

        # Calculate all ma_sell values using HA close
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe['ha_close'], timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['ha_close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe['ha_close'], timeperiod=100)

        dataframe['sma_9'] = ta.SMA(dataframe['ha_close'], timeperiod=9)
        
        # Modify EWO to use HA values
        dataframe['EWO'] = EWO(dataframe.replace({
            'low': dataframe['ha_low'],
            'close': dataframe['ha_close']
        }), self.fast_ewo, self.slow_ewo)

        # RSI calculations using HA close
        dataframe['rsi'] = ta.RSI(dataframe['ha_close'], timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe['ha_close'], timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe['ha_close'], timeperiod=20)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate all indicators directly on the main timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                # Using ha_close from same timeframe
                (dataframe['ha_close'].rolling(self.lookback_candles.value).max()
                 < (dataframe['ha_close'] * self.profit_threshold.value))
            )
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['ha_close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['ha_close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['ha_close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['ha_close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['ha_close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['ha_close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            ((dataframe['ha_close'] > dataframe['sma_9']) &
                (dataframe['ha_close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
             )
            |
            (
                (dataframe['ha_close'] < dataframe['hma_50']) &
                (dataframe['ha_close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
