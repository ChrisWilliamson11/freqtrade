#!/usr/bin/env python3
import os
import glob
import subprocess

# Find all strategy files in the user_data/strategies folder
strategy_dir = "user_data/strategies"
strategy_files = glob.glob(f"{strategy_dir}/[!_]*.py")  # Ignore files starting with _

# Filter out backup files and invalid names
valid_strategies = []
for file in strategy_files:
    name = os.path.splitext(os.path.basename(file))[0]
    # Skip files with spaces, parentheses, or other special characters
    if not any(char in name for char in [' ', '(', ')', 'copy']):
        valid_strategies.append(name)

if not valid_strategies:
    print("No valid strategy files found!")
    exit(1)

# Build the freqtrade command
cmd = ["freqtrade", "backtesting"]
cmd.extend(["--timerange", "20230301-20240101"])
cmd.extend(["--strategy-list"] + valid_strategies)  # Pass each strategy as separate argument
cmd.extend(["--timeframe", "15m"])
cmd.extend(["--dry-run-wallet", "1000"])
cmd.extend(["--stake-amount", "25"])
cmd.extend(["--max-open-trades", "5"])
cmd.extend(["--config", "config.json"])
cmd.extend(["--export", "trades"])
cmd.extend(["--export-filename", "comparison_results.json"])

# Run the command
print(f"Found strategies: {valid_strategies}")
print("Running backtest comparison...")
subprocess.run(cmd) 