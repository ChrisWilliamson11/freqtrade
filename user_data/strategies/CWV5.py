# for live trailing_stop = False and use_custom_stoploss = True
# for backtest trailing_stop = True and use_custom_stoploss = False

# --- Do not remove these libs ---
from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Tuple
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade, LocalTrade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from logging import getLogger
from freqtrade.exchange import timeframe_to_seconds
import requests
from cachetools import TTLCache
from scipy.stats import percentileofscore

logger = getLogger(__name__)

# @Rallipanos
# @pluxury

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 8,
    "ewo_high": 2.403,
    "ewo_high_2": -5.585,
    "ewo_low": -14.378,
    "lookback_candles": 3,
    "low_offset": 0.984,
    "low_offset_2": 0.942,
    "profit_threshold": 1.008,
    "rsi_buy": 72,
    "atr_period": 20  # New parameter
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


class CWV5(IStrategy):
    INTERFACE_VERSION = 2

    # Add protections configuration
    protections = [
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 1,
            "stop_duration": 60,
            "required_profit": -0.05
        },
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,
            "trade_limit": 1,
            "stop_duration_candles": 6,
            "required_profit": -0.01,
            "only_per_pair": True
        }
    ]

    # ROI table:
    minimal_roi = {
        # "0": 0.283,
        # "40": 0.086,
        # "99": 0.036,
        "0": 10
    }

    # Stoploss:
    stoploss = -0.15

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
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.016
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
    timeframe = '5m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = True

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02,
        'liquidity_ratio_threshold': 0.5,  # Minimum ratio of available liquidity to trade size
        'spread_threshold': 0.001,  # Maximum allowed spread (0.1%)
        'max_spread_ratio': 0.75  # Maximum allowed slippage as ratio of spread
    }

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'market',
        'trailing_stop_loss': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }
    # Custom Trailing Stoploss by Perkmeister

    # Add new parameters
    atr_period = IntParameter(
        10, 30, default=buy_params['atr_period'], space='buy', optimize=True)

    # Position sizing params
    risk_per_trade = DecimalParameter(0.01, 0.03, default=0.02, space='buy', decimals=2, optimize=True)
    atr_stop_multiplier = DecimalParameter(2.0, 4.0, default=3.0, space='buy', decimals=1, optimize=True)
    max_pos_size = DecimalParameter(0.1, 0.5, default=0.2, space='buy', decimals=2, optimize=True)  # Max 20% of capital per trade

    # Correlation parameters
    correlation_threshold = DecimalParameter(0.5, 0.9, default=0.7, space='buy', decimals=2, optimize=True)
    correlation_lookback = IntParameter(10, 100, default=50, space='buy', optimize=True)

    # Volatility parameters
    vix_period = IntParameter(10, 30, default=20, space='buy', optimize=True)
    high_vol_threshold = DecimalParameter(0.7, 0.9, default=0.75, space='buy', decimals=2, optimize=True)

    # Regular and volatility-adjusted parameters
    ewo_high_normal = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    ewo_high_volat = DecimalParameter(1.8, 10.8, default=buy_params['ewo_high'] * 0.9, space='buy', optimize=True)
    
    stoploss_normal = DecimalParameter(-0.20, -0.15, default=-0.18, space='sell', optimize=True)
    stoploss_volat = DecimalParameter(-0.15, -0.10, default=-0.12, space='sell', optimize=True)
    
    # Regime-specific EWO parameters
    ewo_low_normal = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_low_volat = DecimalParameter(-18.0, -7.2, default=buy_params['ewo_low'] * 0.9, space='buy', optimize=True)

    # VWAP parameters
    vwap_profit_factor = DecimalParameter(1.01, 1.03, default=1.02, space='sell', decimals=2, optimize=True)
    vwap_window = IntParameter(20, 200, default=100, space='sell', optimize=True)

    # Fibonacci parameters
    fib_window = IntParameter(20, 100, default=50, space='sell', optimize=True)
    fib_profit_factor = DecimalParameter(1.01, 1.05, default=1.02, space='sell', decimals=2, optimize=True)

    # Risk management parameters
    max_drawdown_threshold = DecimalParameter(10.0, 20.0, default=15.0, space='protection', decimals=1, optimize=True)
    recovery_factor = DecimalParameter(0.5, 1.0, default=0.75, space='protection', decimals=2, optimize=True)
    drawdown_cooldown = IntParameter(12, 72, default=24, space='protection', optimize=True)  # Hours
    
    # Store original protection settings
    original_protections = protections.copy()
    original_max_open_trades = None
    last_drawdown_adjust = None
    
    # News filter parameters
    news_filter_enabled = False  # Disable news filter for now
    news_pairs = ['BTC/USDT', 'ETH/USDT']  # Pairs affected by major news
    news_cache = TTLCache(maxsize=100, ttl=3600)  # Cache news data for 1 hour
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.original_max_open_trades = config['max_open_trades']
        self.last_drawdown_adjust = datetime.utcnow() - timedelta(hours=24)  # Initialize with offset

    def get_drawdown_info(self) -> Tuple[float, float, float]:
        """
        Calculate current drawdown metrics
        """
        trades = LocalTrade.get_trades_proxy()
        if not trades:
            return 0.0, 0.0, 0.0
            
        # Calculate equity curve
        equity = self.wallets.get_total_stake_amount()
        starting_balance = self.wallets.get_starting_balance()
        profit_abs = equity - starting_balance
        
        # Calculate running maximum equity
        running_max = max(starting_balance, equity)
        
        # Calculate drawdown
        current_drawdown = ((running_max - equity) / running_max) * 100
        
        # Calculate win rate
        closed_trades = [t for t in trades if not t.is_open]
        if closed_trades:
            win_rate = len([t for t in closed_trades if t.close_profit > 0]) / len(closed_trades) * 100
        else:
            win_rate = 0
            
        return current_drawdown, profit_abs, win_rate

    def adjust_risk_on_drawdown(self, current_time: datetime) -> None:
        """
        Adjust strategy parameters based on drawdown
        """
        # Check if enough time has passed since last adjustment
        if (current_time - self.last_drawdown_adjust).total_seconds() < self.drawdown_cooldown.value * 3600:
            return
            
        drawdown, profit_abs, win_rate = self.get_drawdown_info()
        
        # If drawdown exceeds threshold, implement risk reduction
        if drawdown > self.max_drawdown_threshold.value:
            logger.info(f"Drawdown protection activated: {drawdown:.2f}% drawdown detected")
            
            # Tighten protection settings
            self.protections = [
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": 60,
                    "trade_limit": 1,
                    "stop_duration": 60,
                    "required_profit": -0.02  # Tighter stop
                },
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": 4  # Longer cooldown
                },
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 24,
                    "trade_limit": 1,
                    "stop_duration_candles": 12,
                    "required_profit": -0.01,
                    "only_per_pair": True
                }
            ]
            
            # Reduce max open trades
            self.max_open_trades = max(1, int(self.original_max_open_trades * self.recovery_factor.value))
            
            # Adjust position sizing
            self.risk_per_trade.value = self.risk_per_trade.value * self.recovery_factor.value
            
            self.last_drawdown_adjust = current_time
            
        # If conditions improve, gradually restore original settings
        elif drawdown < self.max_drawdown_threshold.value * 0.5 and win_rate > 50:
            logger.info(f"Restoring normal risk parameters: Drawdown reduced to {drawdown:.2f}%")
            self.protections = self.original_protections.copy()
            self.max_open_trades = self.original_max_open_trades
            self.risk_per_trade.value = self.risk_per_trade.default
            
            self.last_drawdown_adjust = current_time

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, 
                          time_in_force: str, current_time: datetime, **kwargs) -> bool:
        """
        Called before placing a buy order.
        Implement final checks here.
        """
        # Check for news events before entry
        if self.is_news_event_window(current_time, pair):
            logger.info(f"Blocking trade entry for {pair} due to news event")
            return False
            
        # Run existing drawdown protection check
        self.adjust_risk_on_drawdown(current_time)
        
        # Get current drawdown info
        drawdown, profit_abs, win_rate = self.get_drawdown_info()
        
        # Block new entries if drawdown is too severe
        if drawdown > self.max_drawdown_threshold.value * 1.5:
            logger.info(f"Circuit breaker activated: {drawdown:.2f}% drawdown")
            return False
            
        return True

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: float, max_stake: float,
                          **kwargs) -> float:
        """
        Adjust stake amount based on drawdown
        """
        drawdown, profit_abs, win_rate = self.get_drawdown_info()
        
        # Calculate drawdown factor (reduces position size as drawdown increases)
        drawdown_factor = max(0.2, 1 - (drawdown / (self.max_drawdown_threshold.value * 2)))
        
        # Get the base stake calculation
        stake_amount = super().custom_stake_amount(
            pair, current_time, current_rate, proposed_stake, min_stake, max_stake, **kwargs
        )
        
        # Apply drawdown adjustment
        adjusted_stake = stake_amount * drawdown_factor
        
        return max(min_stake, min(adjusted_stake, max_stake))

    def get_regime_parameters(self, dataframe: DataFrame, current_index: int) -> dict:
        """
        Get parameter values based on current market regime
        """
        is_high_vol = dataframe['high_vol'].iloc[current_index] == 1
        
        return {
            'ewo_high': self.ewo_high_volat.value if is_high_vol else self.ewo_high_normal.value,
            'ewo_low': self.ewo_low_volat.value if is_high_vol else self.ewo_low_normal.value,
            'stoploss': self.stoploss_volat.value if is_high_vol else self.stoploss_normal.value,
            'rsi_threshold_adjustment': 0.9 if is_high_vol else 1.0,
            'profit_take_multiplier': 0.8 if is_high_vol else 1.0
        }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # Get dataframe for current pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1]

        # Get minutes elapsed since trade opened
        minutes_elapsed = (current_time - trade.open_date_utc).total_seconds() / 60
        
        # Calculate number of 5-minute periods elapsed
        periods_elapsed = minutes_elapsed / 5
        
        # Get current ATR value
        current_atr = current_candle['atr']
        
        # Calculate volatility-adjusted stoploss increment
        atr_pct = (current_atr / current_rate) * 100
        stoploss_adjustment = periods_elapsed * (atr_pct / 100)
        
        # Calculate new stoploss starting from -0.15
        dynamic_stoploss = self.stoploss + stoploss_adjustment
        
        # Cap at 0 to prevent positive stoploss
        dynamic_stoploss = min(0, dynamic_stoploss)

        # If current profit is worse than our dynamic stoploss, exit immediately
        if current_profit <= dynamic_stoploss:
            return 0.001

        # Enhanced profit-based stoploss logic with parabolic adjustment
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # Parabolic profit protection
        if current_profit >= 0.05:  # Above 5% profit
            # Lock in 50% of profits above 5% plus a parabolic component
            parabolic_factor = (current_profit - 0.05) ** 2  # Squared component for parabolic curve
            sl_profit = max(
                SL_2,  # Don't go below SL_2
                current_profit * 0.5 + parabolic_factor  # Lock 50% plus parabolic component
            )
        elif current_profit >= 0.03:  # Between 3% and 5%
            # Lock in 30% of profits between 3-5% with linear interpolation
            sl_profit = max(
                SL_1,  # Don't go below SL_1
                current_profit * 0.3  # Lock 30% of current profits
            )
        elif current_profit > PF_2:  # Original logic for profits between PF_2 and 3%
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:  # Original logic for profits between PF_1 and PF_2
            sl_profit = SL_1 + ((current_profit - PF_1)*(SL_2 - SL_1)/(PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Add ATR-based buffer to profit-based stoploss
        atr_buffer = (current_atr / current_rate)
        sl_profit = sl_profit - atr_buffer

        # Additional time-based profit protection
        if minutes_elapsed > 720:  # After 12 hours
            sl_profit = max(sl_profit, current_profit * 0.7)  # Lock at least 70% of current profit
        elif minutes_elapsed > 360:  # After 6 hours
            sl_profit = max(sl_profit, current_profit * 0.5)  # Lock at least 50% of current profit

        # Use the more aggressive (higher) stoploss between time-based and profit-based
        return max(dynamic_stoploss, stoploss_from_open(sl_profit, current_profit))

    def analyze_order_book(self, pair: str, stake_amount: float) -> dict:
        """
        Analyze order book for a given pair and stake amount
        """
        try:
            # Get order book
            order_book = self.dp.orderbook(pair, 1)
            if not order_book:
                return {
                    'spread': 0,
                    'liquidity_ratio': 0,
                    'max_slippage': self.slippage_protection['max_slippage']
                }

            # Calculate spread
            best_ask = order_book['asks'][0][0]
            best_bid = order_book['bids'][0][0]
            spread = (best_ask - best_bid) / best_bid

            # Calculate available liquidity
            bid_liquidity = sum(bid[1] * bid[0] for bid in order_book['bids'])
            ask_liquidity = sum(ask[1] * ask[0] for ask in order_book['asks'])
            avg_liquidity = (bid_liquidity + ask_liquidity) / 2

            # Calculate liquidity ratio
            liquidity_ratio = avg_liquidity / stake_amount if stake_amount > 0 else 0

            # Calculate dynamic max slippage based on spread
            max_slippage = min(
                self.slippage_protection['max_slippage'],
                spread * -self.slippage_protection['max_spread_ratio']
            )

            return {
                'spread': spread,
                'liquidity_ratio': liquidity_ratio,
                'max_slippage': max_slippage
            }

        except Exception as e:
            logger.warning(f"Error analyzing order book for {pair}: {e}")
            return {
                'spread': 0,
                'liquidity_ratio': 0,
                'max_slippage': self.slippage_protection['max_slippage']
            }

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # Technical analysis based exit confirmation
        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):
                    return False

        # Enhanced slippage protection
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        # Analyze order book
        order_book_analysis = self.analyze_order_book(pair, trade.stake_amount)
        
        # Check spread
        if order_book_analysis['spread'] > self.slippage_protection['spread_threshold']:
            logger.info(f"Spread too high for {pair}: {order_book_analysis['spread']:.4f}")
            return False

        # Check liquidity
        if order_book_analysis['liquidity_ratio'] < self.slippage_protection['liquidity_ratio_threshold']:
            logger.info(f"Insufficient liquidity for {pair}: {order_book_analysis['liquidity_ratio']:.4f}")
            return False

        # Calculate slippage
        slippage = (rate / last_candle['close']) - 1
        max_slippage = order_book_analysis['max_slippage']

        if slippage < max_slippage:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                logger.info(f"Slippage too high for {pair}: {slippage:.4f}, retry {pair_retries + 1}")
                return False

        # Reset retries
        state[pair] = 0

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        
        # Add ATR calculation
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        
        # Calculate dynamic RSI threshold
        dataframe['dynamic_rsi_threshold'] = 70 + (dataframe['atr'] / dataframe['close'].rolling(50).mean() * 100)
        
        # Add volume trend indicators
        dataframe['volume_ema_20'] = ta.EMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_spike'] = np.where(dataframe['volume'] > (dataframe['volume_ema_20'] * 1.5), 1, 0)
        
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # Volatility state detection
        dataframe['true_range'] = ta.TRANGE(dataframe)
        
        # Calculate volatility index (VIX-like)
        dataframe['vix'] = (
            dataframe['true_range']
            .rolling(self.vix_period.value)
            .mean() * np.sqrt(288)  # Project to daily volatility (288 5-min periods per day)
        )
        
        # Determine high volatility state
        dataframe['vix_percentile'] = (
            dataframe['vix']
            .rolling(100)  # Look back 100 candles for percentile calculation
            .apply(lambda x: percentileofscore(x, x.iloc[-1]))
        )
        
        dataframe['high_vol'] = np.where(
            dataframe['vix_percentile'] > (self.high_vol_threshold.value * 100), 
            1, 
            0
        )

        # Calculate VWAP
        dataframe['volume_price'] = dataframe['volume'] * dataframe['close']
        
        # Rolling VWAP calculation
        dataframe['vwap'] = (
            dataframe['volume_price'].rolling(window=self.vwap_window.value).sum() /
            dataframe['volume'].rolling(window=self.vwap_window.value).sum()
        )
        
        # VWAP-based exit signals
        dataframe['exit_vwap'] = np.where(
            (dataframe['close'] > dataframe['vwap'] * self.vwap_profit_factor.value) &
            (dataframe['volume'] > dataframe['volume_ema_20']),  # Volume confirmation
            1, 
            0
        )
        
        # Calculate volume profile
        dataframe['vol_profile'] = (
            dataframe['volume']
            .rolling(window=self.vwap_window.value)
            .apply(lambda x: np.percentile(x, 70))  # High volume threshold
        )
        
        # Identify high volume resistance levels
        dataframe['vol_resistance'] = np.where(
            (dataframe['volume'] > dataframe['vol_profile']) &
            (dataframe['close'] < dataframe['close'].shift(1)),  # Price rejection
            dataframe['high'],
            np.nan
        )
        
        # Rolling resistance level
        dataframe['resistance_level'] = (
            dataframe['vol_resistance']
            .rolling(window=self.vwap_window.value, min_periods=1)
            .mean()
        )

        # Calculate Fibonacci levels
        dataframe['swing_high'] = dataframe['high'].rolling(self.fib_window.value).max()
        dataframe['swing_low'] = dataframe['low'].rolling(self.fib_window.value).min()
        
        # Calculate Fibonacci retracement levels
        range_price = dataframe['swing_high'] - dataframe['swing_low']
        dataframe['fib_236'] = dataframe['swing_high'] - range_price * 0.236
        dataframe['fib_382'] = dataframe['swing_high'] - range_price * 0.382
        dataframe['fib_500'] = dataframe['swing_high'] - range_price * 0.500
        dataframe['fib_618'] = dataframe['swing_high'] - range_price * 0.618
        
        # Calculate Fibonacci extension levels
        dataframe['fib_127'] = dataframe['swing_high'] + range_price * 0.272
        dataframe['fib_168'] = dataframe['swing_high'] + range_price * 0.618
        
        # Identify price position relative to Fibonacci levels
        dataframe['fib_zone'] = np.where(
            dataframe['close'] > dataframe['fib_168'], 'ext_168',
            np.where(
                dataframe['close'] > dataframe['fib_127'], 'ext_127',
                np.where(
                    dataframe['close'] > dataframe['swing_high'], 'above_high',
                    np.where(
                        dataframe['close'] > dataframe['fib_236'], 'ret_236',
                        np.where(
                            dataframe['close'] > dataframe['fib_382'], 'ret_382',
                            np.where(
                                dataframe['close'] > dataframe['fib_500'], 'ret_500',
                                np.where(
                                    dataframe['close'] > dataframe['fib_618'], 'ret_618',
                                    'below_618'
                                )
                            )
                        )
                    )
                )
            )
        )
        
        # Generate Fibonacci-based profit signals
        dataframe['fib_profit_signal'] = np.where(
            (dataframe['close'] > dataframe['fib_127']) &  # Price above 127% extension
            (dataframe['volume'] > dataframe['volume_ema_20']) &  # Volume confirmation
            (dataframe['rsi'] > 70),  # Overbought condition
            1,
            0
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def get_pair_correlation(self, pair: str, dataframe: DataFrame) -> float:
        """
        Calculate correlation between current pair and existing positions
        """
        try:
            # Get all open trades
            open_trades = Trade.get_trades_proxy(is_open=True)
            
            if not open_trades:
                return 0
            
            # Get current pair's returns
            pair_returns = dataframe['close'].pct_change()
            
            correlations = []
            
            # Calculate correlation with each open position
            for trade in open_trades:
                if trade.pair == pair:
                    continue
                    
                # Get other pair's dataframe
                other_df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                other_returns = other_df['close'].pct_change()
                
                # Calculate correlation over lookback period
                correlation = pair_returns.tail(self.correlation_lookback.value).corr(
                    other_returns.tail(self.correlation_lookback.value)
                )
                correlations.append(abs(correlation))
            
            # Return max correlation with any existing position
            return max(correlations) if correlations else 0
            
        except Exception as e:
            logger.warning(f"Error calculating correlation for {pair}: {e}")
            return 0

    def fetch_economic_calendar(self) -> list:
        """
        Fetch economic calendar events from external API
        Returns list of high impact events
        """
        # Return empty list when disabled
        if not self.news_filter_enabled:
            return []
            
        try:
            # Cache check
            cache_key = datetime.utcnow().strftime('%Y-%m-%d')
            if cache_key in self.news_cache:
                return self.news_cache[cache_key]

            # TODO: Replace with your actual API endpoint and key
            # For now, return empty list to prevent errors
            return []
                
        except Exception as e:
            logger.warning(f"Error fetching economic calendar: {e}")
            return []

    def is_news_event_window(self, current_time: datetime, pair: str) -> bool:
        """
        Check if we're in a high-impact news event window
        """
        if not self.news_filter_enabled or pair not in self.news_pairs:
            return False
            
        return False  # Disabled for now

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dont_buy_conditions = []
        
        # Check for news events
        current_time = datetime.utcnow()
        if self.is_news_event_window(current_time, metadata['pair']):
            dont_buy_conditions.append(
                (
                    # Don't buy during high impact news events
                    (dataframe['volume'] > 0)  # Always true condition
                )
            )

        # Get regime parameters - Simplified approach
        is_high_vol = dataframe['high_vol'] == 1
        rsi_threshold_adjustment = np.where(is_high_vol, 0.9, 1.0)
        ewo_high_adjusted = np.where(is_high_vol, self.ewo_high_volat.value, self.ewo_high_normal.value)
        ewo_low_adjusted = np.where(is_high_vol, self.ewo_low_volat.value, self.ewo_low_normal.value)
        price_offset_factor = np.where(is_high_vol, 0.99, 1.0)

        # Add correlation check to dont_buy_conditions
        correlation = self.get_pair_correlation(metadata['pair'], dataframe)
        if correlation > self.correlation_threshold.value:
            dont_buy_conditions.append(
                (
                    # Don't buy if correlation with existing positions is too high
                    (dataframe['volume'] > 0)  # Always true condition to apply correlation filter
                )
            )

        dont_buy_conditions.append(
            (
                # don't buy if there isn't 3% profit to be made
                (dataframe['close_1h'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        # Add volume condition to dont_buy_conditions
        dont_buy_conditions.append(
            (
                # don't buy if volume is too low compared to average
                (dataframe['volume'] < (dataframe['volume_ema_20'] * 0.5))
            )
        )

        # Add volatility condition to dont_buy_conditions
        dont_buy_conditions.append(
            (
                # Don't buy in high volatility unless there's a strong signal
                (dataframe['high_vol'] == 1) & 
                (
                    (dataframe['volume_spike'] == 0) |  # No volume confirmation
                    (dataframe['rsi'] > dataframe['dynamic_rsi_threshold'] * 0.9)  # RSI not low enough in high vol
                )
            )
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < (35 * rsi_threshold_adjustment)) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * 
                    self.low_offset.value * price_offset_factor)) &
                (dataframe['EWO'] > ewo_high_adjusted) &
                (dataframe['rsi'] < dataframe['dynamic_rsi_threshold']) &
                (dataframe['volume'] > 0) &
                (dataframe['volume_spike'] == 1) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < (35 * rsi_threshold_adjustment)) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * 
                    self.low_offset_2.value * price_offset_factor)) &
                (dataframe['EWO'] > ewo_high_adjusted) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['volume_spike'] == 1) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < (35 * rsi_threshold_adjustment)) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * 
                    self.low_offset.value * price_offset_factor)) &
                (dataframe['EWO'] < ewo_low_adjusted) &
                (dataframe['volume'] > 0) &
                (dataframe['volume_spike'] == 1) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Get regime parameters - Modify this part
        is_high_vol = dataframe['high_vol'] == 1
        profit_take_multiplier = np.where(is_high_vol, 0.8, 1.0)

        # VWAP-based exit condition
        conditions.append(
            (dataframe['exit_vwap'] == 1) &
            (dataframe['volume'] > dataframe['volume_ema_20'] * 1.2) &
            (
                (dataframe['close'] > dataframe['resistance_level']) |
                (dataframe['high_vol'] == 1)
            )
        )

        # Original conditions with volume profile enhancement - Modified
        conditions.append(
            ((dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * 
                    self.high_offset_2.value * profit_take_multiplier)) &  # Modified this line
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > dataframe['volume_ema_20']) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow']) &
                (dataframe['close'] > dataframe['vwap'])
             )
            |
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > dataframe['volume_ema_20']) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )
        )

        # Rest of the conditions remain the same
        conditions.append(
            (dataframe['close'] > dataframe['resistance_level']) &
            (dataframe['volume'] > dataframe['vol_profile']) &
            (dataframe['rsi'] > 70) &
            (dataframe['close'] > dataframe['close'].shift(1) * 1.02)
        )

        # Fibonacci-based exit conditions
        conditions.append(
            (dataframe['fib_profit_signal'] == 1) &
            (
                (dataframe['close'] > dataframe['fib_127'] * self.fib_profit_factor.value) |
                (
                    (dataframe['close'] > dataframe['fib_168']) &
                    (dataframe['volume'] > dataframe['volume_ema_20'] * 1.5)
                )
            )
        )

        # Fibonacci retracement protection
        conditions.append(
            (dataframe['close'] < dataframe['fib_382']) &
            (dataframe['volume'] > dataframe['volume_ema_20']) &
            (dataframe['rsi'] < 30) &
            (dataframe['high_vol'] == 1)
        )

        # Fibonacci stoploss calculation
        dataframe['fib_stoploss'] = np.where(
            dataframe['fib_zone'] == 'ext_168', dataframe['fib_127'],
            np.where(
                dataframe['fib_zone'] == 'ext_127', dataframe['swing_high'],
                np.where(
                    dataframe['fib_zone'] == 'above_high', dataframe['fib_236'],
                    dataframe['fib_618']
                )
            )
        )

        # Add trailing stop condition
        conditions.append(
            (dataframe['close'] < dataframe['fib_stoploss']) &
            (dataframe['volume'] > dataframe['volume_ema_20'])
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: float, max_stake: float,
                            **kwargs) -> Optional[float]:
        """
        Adjusts position size for open trades based on market conditions
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        current_candle = dataframe.iloc[-1]
        
        # Don't increase position during news events
        if self.is_news_event_window(current_time, trade.pair):
            return None
        
        # Don't increase position in high volatility
        if current_candle['high_vol'] == 1:
            return None
        
        # Check correlation before increasing position
        correlation = self.get_pair_correlation(trade.pair, dataframe)
        if correlation > self.correlation_threshold.value:
            return None  # Don't increase position if correlation is too high
        
        # Check if we should increase position
        if current_profit > 0.02 and current_candle['volume_spike'] == 1:
            total_capital = self.wallets.get_total_stake_amount()
            current_stake = trade.stake_amount
            
            additional_stake = current_stake * 0.5
            
            total_stake = current_stake + additional_stake
            if total_stake <= (total_capital * self.max_pos_size.value):
                return additional_stake
        
        return None
