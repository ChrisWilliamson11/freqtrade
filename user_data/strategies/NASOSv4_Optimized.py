# NASOSv4 Optimized Strategy with Enhanced Profit Protection
# Generated: Sunday, February 16, 2025
# Timeframe: 5min | Capital: $2,000

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from typing import Dict, List, Optional
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from functools import reduce

# Enhanced buy parameters with volatility scaling
buy_params = {
    "base_nb_candles_buy": 14,  # Increased from 8 for better trend confirmation
    "ewo_low": -16.5,  # Expanded from -14.378
    "ewo_high": 3.2,   # From 2.403
    "rsi_buy": 68      # Reduced from 72 for earlier entries
}

# Aggressive but protected sell parameters
sell_params = {
    "base_nb_candles_sell": 22,  # From 16
    "high_offset": 1.12,         # From 1.084
    "pPF_1": 0.024,             # Increased from 0.016
    "pSL_1": 0.020              # From 0.014
}

class NASOSv4_Optimized(IStrategy):
    
    # Volatility adaptive configuration
    timeframe = '5m'
    inf_1h = '1h'
    minimal_roi = {
        "0": 0.15,    # Reduced from 0.18
        "30": 0.08,   # Increased from 0.06
        "60": 0.04,   # Increased from 0.03
        "180": 0.02   # Added longer-term hold for trending moves
    }
    stoploss = -0.18
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Strategy parameters
    ewo_high = DecimalParameter(-20.0, 20.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    ewo_low = DecimalParameter(-20.0, 1.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)
    
    # Enhanced protections
    protections = [
        {   # Dynamic drawdown protection
            "method": "LowProfitPairs",
            "lookback_period_candles": 72,
            "trade_limit": 2,
            "stop_duration": 48,
            "required_profit": -0.08
        },
        {   # Volatility cooldown
            "method": "CooldownPeriod",
            "stop_duration_candles": 4  # Fixed value for cooldown period
        }
    ]

    # Hyperparameters with wider ranges for dynamic adjustment
    base_nb_candles_buy = IntParameter(8, 20, default=14, space='buy')
    low_offset = DecimalParameter(0.92, 0.98, default=0.955, space='buy')
    high_offset = DecimalParameter(1.05, 1.2, default=1.12, space='sell')

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.account_balance = config['dry_run_wallet']
        self.volatility_state = 'medium'
        self.last_regime_shift = datetime.now()
        self.high_vol = False
        # Track pair performance
        self.pair_performance = {}
        self.pair_trades = {}
        # Track trailing stops
        self.custom_trailing_stops = {}
        self.trailing_stop_multipliers = {
            'BTC/USDT': 1.2,  # More room for major pairs
            'ETH/USDT': 1.2,
            'SOL/USDT': 0.8,  # Tighter for volatile pairs
            'AVAX/USDT': 0.8
        }

    def informative_pairs(self) -> List[tuple]:
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '15m') for pair in pairs]
        informative_pairs.extend([(pair, '1h') for pair in pairs])
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators required for the strategies."""
        # Get higher timeframe data
        inf_tf2 = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='15m')
        inf_tf3 = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
        
        # Higher timeframe trend with enhanced confirmation
        inf_tf2['ema_50'] = ta.EMA(inf_tf2, timeperiod=50)
        inf_tf2['ema_200'] = ta.EMA(inf_tf2, timeperiod=200)
        inf_tf2['trend_long'] = inf_tf2['ema_50'] > inf_tf2['ema_200']
        inf_tf2['trend_strength'] = (inf_tf2['ema_50'] - inf_tf2['ema_200']) / inf_tf2['ema_200'] * 100
        dataframe['15m_trend'] = inf_tf2['trend_long'].astype(int)
        dataframe['15m_trend_strength'] = inf_tf2['trend_strength']
        
        inf_tf3['ema_50'] = ta.EMA(inf_tf3, timeperiod=50)
        inf_tf3['ema_200'] = ta.EMA(inf_tf3, timeperiod=200)
        inf_tf3['trend_long'] = inf_tf3['ema_50'] > inf_tf3['ema_200']
        inf_tf3['trend_strength'] = (inf_tf3['ema_50'] - inf_tf3['ema_200']) / inf_tf3['ema_200'] * 100
        dataframe['1h_trend'] = inf_tf3['trend_long'].astype(int)
        dataframe['1h_trend_strength'] = inf_tf3['trend_strength']
        
        # Enhanced volume analysis with adaptive thresholds
        dataframe['volume_mean'] = ta.SMA(dataframe['volume'], timeperiod=200)
        dataframe['volume_std'] = dataframe['volume'].rolling(200).std()
        dataframe['volume_z_score'] = (dataframe['volume'] - dataframe['volume_mean']) / dataframe['volume_std']
        
        # Dynamic volume thresholds with pair-specific adjustments
        vol_multiplier = 1.5
        vol_z_score_req = 0.8
        if metadata['pair'] in ['BTC/USDT', 'ETH/USDT']:
            vol_multiplier = 1.2  # More lenient for major pairs
            vol_z_score_req = 0.6
        elif metadata['pair'] in ['SOL/USDT', 'AVAX/USDT']:
            vol_multiplier = 1.8  # Stricter for volatile pairs
            vol_z_score_req = 1.0
            
        dataframe['high_volume_threshold'] = dataframe['volume_mean'] + (dataframe['volume_std'] * vol_multiplier)
        dataframe['very_high_volume'] = dataframe['volume'] > dataframe['high_volume_threshold']
        
        # Enhanced momentum with multiple timeframes
        dataframe['macd'], dataframe['macdsignal'], _ = ta.MACD(dataframe['close'])
        dataframe['macd_cross_up'] = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        dataframe['macd_above'] = dataframe['macd'] > dataframe['macdsignal']
        dataframe['macd_increasing'] = dataframe['macd'] > dataframe['macd'].shift(1)
        
        # RSI with dynamic thresholds
        for timeperiod in [14, 30, 100]:
            dataframe[f'rsi_{timeperiod}'] = ta.RSI(dataframe, timeperiod=timeperiod)
            
        # Trend strength with multiple confirmations
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        dataframe['trend_long'] = (
            (dataframe['ema_50'] > dataframe['ema_200']) &
            (dataframe['ema_20'] > dataframe['ema_50']) &
            (dataframe['ema_20'].shift(1) > dataframe['ema_20'].shift(2))  # Ensuring upward momentum
        )
        
        dataframe['trend_strength'] = (dataframe['ema_50'] - dataframe['ema_200']) / dataframe['ema_200'] * 100
        
        # Enhanced volatility metrics with pair-specific adjustments
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['natr'] = ta.NATR(dataframe, timeperiod=14)
        
        # Pair-specific volatility thresholds
        high_vol_threshold = 2.0
        low_vol_threshold = 1.0
        if metadata['pair'] in ['BTC/USDT', 'ETH/USDT']:
            high_vol_threshold = 1.8
            low_vol_threshold = 0.8
        elif metadata['pair'] in ['SOL/USDT', 'AVAX/USDT']:
            high_vol_threshold = 2.2
            low_vol_threshold = 1.2
            
        dataframe['volatility_state'] = np.where(
            dataframe['natr'] > high_vol_threshold, 'high',
            np.where(dataframe['natr'] < low_vol_threshold, 'low', 'medium')
        )
        
        # Price action and support/resistance with dynamic zones
        dataframe['pivot'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['atr_support_zone'] = dataframe['pivot'] - (dataframe['atr'] * 0.5)
        dataframe['atr_resistance_zone'] = dataframe['pivot'] + (dataframe['atr'] * 0.5)
        
        # Enhanced support/resistance zones
        dataframe['near_support'] = (
            (dataframe['close'] > dataframe['atr_support_zone']) & 
            (dataframe['close'] < dataframe['atr_support_zone'] * 1.01)
        )
        
        dataframe['near_resistance'] = (
            (dataframe['close'] < dataframe['atr_resistance_zone']) & 
            (dataframe['close'] > dataframe['atr_resistance_zone'] * 0.99)
        )
        
        # Liquidity analysis
        dataframe['vwap'] = qtpylib.rolling_vwap(dataframe, window=20)
        dataframe['liquidity_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        dataframe['high_liquidity'] = (
            (dataframe['liquidity_ratio'] > 1.2) &
            (dataframe['volume'] > dataframe['volume_mean']) &
            (dataframe['volume_z_score'] > 0.5)
        )
        
        # Trend strength indicators
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dmi_plus'] = ta.PLUS_DI(dataframe)
        dataframe['dmi_minus'] = ta.MINUS_DI(dataframe)
        dataframe['strong_trend'] = (
            (dataframe['adx'] > 25) &
            (dataframe['dmi_plus'] > dataframe['dmi_minus']) &
            (dataframe['trend_strength'] > 0)
        )
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enhanced entry logic with adaptive thresholds"""
        conditions = []
        
        # Pair-specific trend strength requirements
        trend_strength_req = 1.0
        if metadata['pair'] in ['BTC/USDT', 'ETH/USDT']:
            trend_strength_req = 0.6  # More lenient for major pairs
        elif metadata['pair'] in ['SOL/USDT', 'AVAX/USDT']:
            trend_strength_req = 1.4  # Stricter for volatile pairs
        
        # Enhanced trend following condition
        trend_condition = (
            dataframe['trend_long'] &
            (dataframe['15m_trend'] > 0) &
            (dataframe['1h_trend'] > 0) &
            (dataframe['trend_strength'] > trend_strength_req) &
            (dataframe['15m_trend_strength'] > trend_strength_req * 0.8) &  # Confirm on higher timeframe
            (dataframe['1h_trend_strength'] > trend_strength_req * 0.5) &   # Confirm on higher timeframe
            (dataframe['ema_20'] > dataframe['ema_50']) &
            (dataframe['close'] > dataframe['vwap'])
        )
        
        # Enhanced volume condition with adaptive thresholds
        volume_condition = (
            dataframe['high_liquidity'] &
            (dataframe['volume'] > dataframe['volume_mean']) &
            (dataframe['volume_z_score'] > 0.8) &  # Base requirement
            (
                (dataframe['volume_z_score'] > 1.2) |  # Either very high volume
                (  # Or consistent volume with trend
                    (dataframe['volume_z_score'] > 0.6) &
                    (dataframe['volume'] > dataframe['volume'].shift(1)) &
                    (dataframe['volume'].shift(1) > dataframe['volume'].shift(2))
                )
            )
        )
        
        # Enhanced momentum condition
        momentum_condition = (
            dataframe['macd_above'] &
            dataframe['macd_increasing'] &
            (dataframe['rsi_14'] < 75) &
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1)) &  # RSI increasing
            (dataframe['rsi_100'] > 40)  # Long-term momentum check
        )
        
        # Strong trend entry with enhanced confirmation
        conditions.append(
            trend_condition &
            volume_condition &
            momentum_condition &
            (dataframe['close'] > dataframe['open']) &
            (dataframe['close'] > dataframe['close'].shift(1)) &
            ~dataframe['near_resistance']
        )
        
        # Counter-trend bounce entry with stricter requirements
        bounce_condition = (
            (dataframe['rsi_14'] < 35) &
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1)) &  # RSI turning up
            (dataframe['rsi_100'] > 35) &  # Not too weak long-term
            dataframe['near_support'] &
            (dataframe['volume_z_score'] > 1.2) &
            (dataframe['macd'] > dataframe['macd'].shift(1)) &  # MACD improving
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['volatility_state'] != 'high')
        )
        
        conditions.append(bounce_condition)
        
        dataframe.loc[
            reduce(lambda x, y: x | y, conditions),
            'buy'
        ] = 1
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """Dynamic stoploss with volatility adaptation"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        
        # Initialize custom trailing stop if not exists
        if pair not in self.custom_trailing_stops:
            self.custom_trailing_stops[pair] = {
                'highest_profit': current_profit,
                'trailing_stop': self.stoploss
            }
        
        # Update highest seen profit
        if current_profit > self.custom_trailing_stops[pair]['highest_profit']:
            self.custom_trailing_stops[pair]['highest_profit'] = current_profit
        
        # Get pair-specific multiplier
        multiplier = self.trailing_stop_multipliers.get(pair, 1.0)
        
        # Dynamic volatility adjustment
        atr = last_candle['atr']
        natr = last_candle['natr']
        
        # Time-based decay
        hours_open = (current_time - trade.open_date_utc).total_seconds() / 3600
        time_factor = max(0, 1 - (hours_open / 48))
        
        # Calculate dynamic stoploss based on volatility and profit
        if current_profit > 0:
            # Profit zone trailing stop
            profit_threshold = 0.02 * multiplier
            if current_profit > profit_threshold:
                # Tighten stop more aggressively as profit increases
                dynamic_sl = max(
                    -0.15 + (current_profit * 0.8),  # More aggressive trailing
                    self.custom_trailing_stops[pair]['trailing_stop']  # Never widen
                )
                # Add volatility scaling
                dynamic_sl += (natr * 0.2 * multiplier)
            else:
                # Normal trailing in profit
                dynamic_sl = -0.15 + (current_profit * 0.5)
                dynamic_sl += (natr * 0.3 * multiplier)
        else:
            # Loss zone trailing stop
            dynamic_sl = self.stoploss + (natr * 0.4 * multiplier)
            
            # Add extra protection in high volatility
            if last_candle['volatility_state'] == 'high':
                dynamic_sl = max(dynamic_sl, self.stoploss * 0.8)
        
        # Never widen existing trailing stop
        if dynamic_sl > self.custom_trailing_stops[pair]['trailing_stop']:
            dynamic_sl = self.custom_trailing_stops[pair]['trailing_stop']
        else:
            self.custom_trailing_stops[pair]['trailing_stop'] = dynamic_sl
        
        return dynamic_sl

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enhanced exit logic with dynamic profit taking"""
        
        # Pair-specific exit thresholds
        rsi_high = 75
        volume_req = 1.3
        if metadata['pair'] in ['BTC/USDT', 'ETH/USDT']:
            rsi_high = 80
            volume_req = 1.2
        elif metadata['pair'] in ['SOL/USDT', 'AVAX/USDT']:
            rsi_high = 72
            volume_req = 1.4
        
        # Dynamic profit taking levels based on volatility
        profit_threshold = dataframe['natr'] * 0.5  # Scale with volatility
        
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['atr_support_zone']) &
                (dataframe['volume'] > dataframe['volume_mean'] * volume_req) &
                (dataframe['close'] < dataframe['close'].shift(1)) &
                (
                    (dataframe['adx'] > 20) |  # Strong trend
                    (dataframe['volatility_state'] == 'high')  # High volatility protection
                )
            ) |
            (
                (dataframe['close'] > dataframe['atr_resistance_zone']) &
                (dataframe['volume'] > dataframe['volume_mean'] * volume_req) &
                (dataframe['rsi_14'] > rsi_high) &
                ~dataframe['macd_above'] &  # MACD bearish
                (
                    (dataframe['close'] > dataframe['close'].shift(1) * (1 + profit_threshold)) |  # Profit target hit
                    (dataframe['volatility_state'] == 'high')  # High volatility protection
                )
            ) |
            (
                (dataframe['close'] < dataframe['ema_20']) &
                (dataframe['close'] < dataframe['close'].shift(1)) &
                ~dataframe['high_liquidity'] &
                (dataframe['volume'] > dataframe['volume_mean']) &
                (
                    (dataframe['adx'] > 25) |  # Strong trend
                    (dataframe['macd'] < dataframe['macd'].shift(3))  # Deteriorating momentum
                )
            ),
            'sell'
        ] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        """Enhanced exit with improved profit protection"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return None
            
        last_candle = dataframe.iloc[-1]
        
        # Update pair performance tracking
        if pair not in self.pair_performance:
            self.pair_performance[pair] = []
        if pair not in self.pair_trades:
            self.pair_trades[pair] = 0
            
        self.pair_trades[pair] += 1
        
        # Calculate pair-specific metrics
        avg_profit = np.mean(self.pair_performance[pair]) if self.pair_performance[pair] else 0
        trade_count = self.pair_trades[pair]
        
        # Dynamic profit threshold based on volatility and performance
        base_profit_threshold = max(0.015, last_candle['natr'] * 0.5)
        if avg_profit > 0:
            base_profit_threshold = max(0.012, avg_profit * 0.7)
        elif avg_profit < 0:
            base_profit_threshold = max(0.018, last_candle['natr'] * 0.6)
            
        # Adjust thresholds based on pair characteristics
        if pair in ['BTC/USDT', 'ETH/USDT']:
            base_profit_threshold *= 0.9  # More conservative for major pairs
        elif pair in ['SOL/USDT', 'AVAX/USDT']:
            base_profit_threshold *= 1.2  # More aggressive for volatile pairs
            
        # Dynamic volatility check
        self.high_vol = last_candle['volatility_state'] == 'high'
        
        # Enhanced loss exit with volatility adaptation
        base_loss_threshold = -0.008 if self.high_vol else -0.012
        if trade_count > 3:
            base_loss_threshold = base_loss_threshold * (1 + (avg_profit * 2))
            
        # Quick loss exit with momentum check
        if current_profit < base_loss_threshold:
            if (
                last_candle['rsi_14'] < 45 or
                (last_candle['macd'] < last_candle['macd'].shift(3)) or
                (last_candle['volatility_state'] == 'high')
            ):
                self.pair_performance[pair].append(current_profit)
                return 'quick_loss_exit'
        
        # Enhanced profit protection with multiple conditions
        if current_profit > base_profit_threshold:
            exit_signals = 0
            if last_candle['rsi_14'] > 72:
                exit_signals += 1
            if not last_candle['macd_above']:
                exit_signals += 1
            if self.high_vol:
                exit_signals += 1
            if last_candle['volume'] < last_candle['volume_mean'] * 0.7:
                exit_signals += 1
                
            if exit_signals >= 2:  # Require multiple confirmation
                self.pair_performance[pair].append(current_profit)
                return 'profit_protection'
        
        # Improved liquidity-based exit with volatility check
        liquidity_threshold = 0.8
        if self.high_vol:
            liquidity_threshold = 0.9  # Stricter in high volatility
            
        if last_candle['liquidity_ratio'] < liquidity_threshold:
            if current_profit > 0.008:
                self.pair_performance[pair].append(current_profit)
                return 'low_liquidity_profit'
            elif current_profit < -0.004:
                self.pair_performance[pair].append(current_profit)
                return 'low_liquidity_loss'
        
        # Enhanced volatility protection with trend confirmation
        if self.high_vol:
            profit_threshold = 0.008 * (1.2 if pair in ['SOL/USDT', 'AVAX/USDT'] else 1.0)
            loss_threshold = -0.006 * (1.2 if pair in ['SOL/USDT', 'AVAX/USDT'] else 1.0)
            
            if current_profit > profit_threshold:
                if not last_candle['trend_long'] or last_candle['rsi_14'] > 70:
                    self.pair_performance[pair].append(current_profit)
                    return 'high_volatility_profit_protection'
            elif current_profit < loss_threshold:
                if not last_candle['trend_long'] or last_candle['rsi_14'] < 40:
                    self.pair_performance[pair].append(current_profit)
                    return 'high_volatility_loss_protection'
        
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """Dynamic position sizing based on volatility and pair performance"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return 1.0
            
        last_candle = dataframe.iloc[-1]
        
        # Get pair performance metrics
        avg_profit = np.mean(self.pair_performance.get(pair, [0]))
        trade_count = self.pair_trades.get(pair, 0)
        
        # Base position size on volatility
        vol_factor = 1.0
        if last_candle['volatility_state'] == 'high':
            vol_factor = 0.7
        elif last_candle['volatility_state'] == 'low':
            vol_factor = 1.2
            
        # Adjust for pair performance
        perf_factor = 1.0
        if trade_count > 3:  # Enough data for adjustment
            if avg_profit > 0:
                perf_factor = min(1.3, 1 + (avg_profit * 2))
            else:
                perf_factor = max(0.7, 1 + (avg_profit * 2))
                
        final_leverage = min(proposed_leverage * vol_factor * perf_factor, max_leverage)
        return round(final_leverage, 2)
