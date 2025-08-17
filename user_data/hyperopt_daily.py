import logging
import json
import datetime
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any
import talib.abstract as ta
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.enums import RunMode
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

# Add the freqtrade root directory to Python path for strategy imports
sys.path.append(str(Path.cwd()))

logger = logging.getLogger(__name__)

class DailyHyperopt:
    def __init__(self, strategy_name: str = "NFI5MOHO_Dynamic"):
        self.strategy_name = strategy_name
        self.config_path = f"user_data/configs/config{strategy_name}.json"
        self.hyperopt_config_path = f"user_data/configs/config{strategy_name}_hyperopt.json"
        self.strategy_path = f"user_data/strategies/{strategy_name}.py"
        self.results_path = "user_data/hyperopt_results"
        Path(self.results_path).mkdir(parents=True, exist_ok=True)

    def analyze_market_conditions(self, timeframe='5m', lookback_days=5) -> str:
        """
        Analyze recent market conditions using multiple indicators
        Returns: 'trending', 'ranging', or 'volatile'
        """
        # Load config to get pairs
        with open(self.hyperopt_config_path, 'r') as f:
            config = json.load(f)
        pairs = config.get('pairs', ["BTC/USDT", "ETH/USDT"])  # Use pairs from config
        
        # Calculate dates ensuring end is after start and in a valid range
        current_date = datetime.datetime.now()
        # Start from 5 days ago and end yesterday to ensure we have data
        end_date = (current_date - datetime.timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
        start_date = (end_date - datetime.timedelta(days=lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        logger.info(f"Downloading data for pairs: {pairs}")
        logger.info(f"Timerange: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
        
        cmd = [
            "freqtrade", "download-data",
            "--exchange", "binance",
            "--pairs", *pairs,  # Unpack all pairs from config
            "--timeframe", timeframe,
            "--timerange", f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}",
            "--config", self.hyperopt_config_path,  # Use hyperopt config
            "--trading-mode", "spot",
            "--erase"  # Erase existing data to ensure clean download
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading data: {e}")
            logger.error("Defaulting to ranging market condition")
            return "ranging"

        # Load the downloaded data
        data_path = Path("user_data/data/binance")
        dfs = []
        for pair in pairs:
            pair_file = data_path / f"binance-{pair.replace('/', '')}-{timeframe}.json"
            if pair_file.exists():
                try:
                    df = pd.read_json(pair_file)
                    if not df.empty:
                        # Convert timestamp to datetime and ensure it's within bounds
                        df['date'] = pd.to_datetime(df['date'], unit='ms')
                        # Filter out any dates beyond our timerange
                        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                        if not df.empty:
                            logger.info(f"Loaded {len(df)} candles for {pair}")
                            dfs.append(df)
                        else:
                            logger.warning(f"No valid data within timerange for {pair}")
                    else:
                        logger.warning(f"Empty dataframe for {pair}")
                except Exception as e:
                    logger.error(f"Error reading data file {pair_file}: {e}")

        if not dfs:
            logger.error("No valid data found for market analysis")
            return "ranging"  # Default to ranging if no data

        try:
            # Combine data from all pairs
            df = pd.concat(dfs)
            df = df.sort_values('date')
            
            logger.info(f"Analyzing {len(df)} total candles from {df['date'].min()} to {df['date'].max()}")
            
            # Calculate market condition indicators
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].std() * np.sqrt(288)  # Annualized volatility
            
            # Calculate ADX for trend strength
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            adx = ta.ADX(df)
            current_adx = adx.iloc[-1] if not adx.empty else 0

            # Calculate price movement
            total_movement = abs(df['close'].iloc[-1] - df['close'].iloc[0])
            price_range = df['high'].max() - df['low'].min()
            directional_efficiency = total_movement / price_range if price_range > 0 else 0

            # Determine market condition using multiple factors
            if current_adx > 25:  # Strong trend
                if volatility > 0.8:  # High volatility
                    logger.info(f"Detected volatile market (ADX: {current_adx:.1f}, Volatility: {volatility:.2f})")
                    return 'volatile'
                logger.info(f"Detected trending market (ADX: {current_adx:.1f}, Volatility: {volatility:.2f})")
                return 'trending'
            else:  # Weak trend
                if volatility > 0.8:  # High volatility
                    logger.info(f"Detected volatile market (ADX: {current_adx:.1f}, Volatility: {volatility:.2f})")
                    return 'volatile'
                if directional_efficiency < 0.3:  # Low directional movement
                    logger.info(f"Detected ranging market (ADX: {current_adx:.1f}, DE: {directional_efficiency:.2f})")
                    return 'ranging'
                logger.info(f"Detected trending market (ADX: {current_adx:.1f}, DE: {directional_efficiency:.2f})")
                return 'trending'
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return "ranging"

    def generate_hyperopt_settings(self, market_condition: str) -> Dict[str, Any]:
        """
        Generate hyperopt settings based on market condition
        """
        # Base settings
        settings = {
            "max_open_trades": 4,
            "timeframe": "5m",
            "timerange": "20230101-",  # Will be updated in run_hyperopt
            "spaces": ["buy", "sell"],
            "epochs": 1000,
            "parallel_jobs": -1,  # Use all CPU cores
            "loss": "ShortTradeDurHyperOptLoss"  # Default loss function
        }

        # Adjust settings based on market condition
        if market_condition == 'trending':
            settings.update({
                "loss": "SharpeHyperOptLoss",  # Focus on risk-adjusted returns
                "epochs": 1500  # More epochs for trend following
            })
        elif market_condition == 'ranging':
            settings.update({
                "loss": "SortinoHyperOptLoss",  # Focus on downside risk
                "epochs": 1200
            })
        elif market_condition == 'volatile':
            settings.update({
                "loss": "MaxDrawDownHyperOptLoss",  # Focus on minimizing drawdown
                "epochs": 1500
            })

        return settings

    def run_hyperopt(self, market_condition: str) -> Dict[str, Any]:
        """
        Run hyperopt with settings optimized for the current market condition
        """
        settings = self.generate_hyperopt_settings(market_condition)
        
        # Set timerange for recent data
        end_date = datetime.datetime.now()
        # Since we're in 2025, use data from the last 30 days before today
        end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=30)
        settings['timerange'] = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
        
        logger.info(f"Running hyperopt for timerange: {settings['timerange']}")

        # Prepare hyperopt command
        cmd = [
            "freqtrade", "hyperopt",
            "--config", self.hyperopt_config_path,  # Use the hyperopt-specific config
            "--strategy", self.strategy_name,
            "--hyperopt-loss", settings['loss'],
            "--spaces", *settings['spaces'],
            "--timerange", settings['timerange'],
            "--epochs", str(settings['epochs']),
            "-j", "1",  # Run single-threaded
            "--no-color",
            "--print-json"
        ]

        logger.info(f"Running hyperopt with command: {' '.join(cmd)}")

        # Run hyperopt
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the JSON output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.strip().startswith('{') and '"params"' in line:
                    hyperopt_results = json.loads(line)
                    return hyperopt_results['params']
            
            logger.error("No valid hyperopt results found in output")
            logger.debug(f"Hyperopt output: {result.stdout}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Hyperopt failed: {e}")
            logger.error(f"Command output: {e.stdout if e.stdout else ''}")
            logger.error(f"Command error: {e.stderr if e.stderr else ''}")
            return None

    def update_strategy_parameters(self, params: Dict[str, Any], market_condition: str):
        """
        Update strategy parameters with hyperopt results
        """
        if not params:
            logger.error("No parameters to update")
            return

        # Save parameters with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(self.results_path) / f"hyperopt_results_{timestamp}.json"
        
        results = {
            "timestamp": timestamp,
            "market_condition": market_condition,
            "parameters": params
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Update strategy file
        try:
            with open(self.strategy_path, 'r') as f:
                strategy_code = f.read()

            # Update buy parameters
            buy_params_start = strategy_code.find("buy_params = {")
            buy_params_end = strategy_code.find("}", buy_params_start) + 1
            
            new_buy_params = "buy_params = {\n"
            for key, value in params.items():
                if key.startswith('buy_'):
                    new_buy_params += f"    '{key}': {value},\n"
            new_buy_params += "}"
            
            strategy_code = (
                strategy_code[:buy_params_start] +
                new_buy_params +
                strategy_code[buy_params_end:]
            )

            # Update sell parameters
            sell_params_start = strategy_code.find("sell_params = {")
            sell_params_end = strategy_code.find("}", sell_params_start) + 1
            
            new_sell_params = "sell_params = {\n"
            for key, value in params.items():
                if key.startswith('sell_'):
                    new_sell_params += f"    '{key}': {value},\n"
            new_sell_params += "}"
            
            strategy_code = (
                strategy_code[:sell_params_start] +
                new_sell_params +
                strategy_code[sell_params_end:]
            )

            # Save updated strategy
            with open(self.strategy_path, 'w') as f:
                f.write(strategy_code)

            logger.info(f"Strategy parameters updated successfully. Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error updating strategy parameters: {e}")

    def run_daily_optimization(self):
        """
        Main function to run daily optimization
        """
        logger.info("Starting daily optimization process...")
        
        # Analyze market conditions
        market_condition = self.analyze_market_conditions()
        logger.info(f"Current market condition: {market_condition}")
        
        # Run hyperopt
        params = self.run_hyperopt(market_condition)
        if params:
            # Update strategy
            self.update_strategy_parameters(params, market_condition)
            logger.info("Daily optimization completed successfully")
        else:
            logger.error("Daily optimization failed - no parameters returned from hyperopt")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("user_data/hyperopt_daily.log")
        ]
    )

    # Run optimization
    optimizer = DailyHyperopt()
    optimizer.run_daily_optimization() 