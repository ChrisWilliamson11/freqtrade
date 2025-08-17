import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import streamlit as st

class FreqtradeDashboard:
    def __init__(self):
        self.trades_file = Path('user_data/backtest_results/latest_backtest.json')
        self.sectors = {
            'Large Caps': ['BTC/USDT', 'ETH/USDT'],
            'Layer 1': ['SOL/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'NEAR/USDT', 'ATOM/USDT'],
            'Layer 2': ['MATIC/USDT', 'ARB/USDT', 'OP/USDT'],
            'DeFi': ['UNI/USDT', 'AAVE/USDT', 'SNX/USDT', 'CAKE/USDT', 'CRV/USDT', 'COMP/USDT'],
            'Gaming': ['SAND/USDT', 'MANA/USDT', 'IMX/USDT', 'AXS/USDT', 'GALA/USDT'],
            'AI & Data': ['OCEAN/USDT', 'FET/USDT', 'AGIX/USDT'],
            'Infrastructure': ['LINK/USDT', 'GRT/USDT', 'RNDR/USDT']
        }

    def load_trades(self):
        with open(self.trades_file) as f:
            data = json.load(f)
        return pd.DataFrame(data['trades'])

    def get_sector_for_pair(self, pair):
        for sector, pairs in self.sectors.items():
            if pair in pairs:
                return sector
        return 'Other'

    def create_dashboard(self):
        st.title('Freqtrade Trading Dashboard')

        # Load trades
        trades_df = self.load_trades()
        trades_df['sector'] = trades_df['pair'].apply(self.get_sector_for_pair)

        # Sector Performance
        st.header('Sector Performance')
        sector_stats = trades_df.groupby('sector').agg({
            'profit_ratio': ['mean', 'sum', 'count'],
            'trade_duration': 'mean'
        }).round(3)
        st.dataframe(sector_stats)

        # Profit Distribution
        st.header('Profit Distribution by Sector')
        fig = px.box(trades_df, x='sector', y='profit_ratio')
        st.plotly_chart(fig)

        # Win Rate by Sector
        st.header('Win Rate by Sector')
        wins_by_sector = trades_df.groupby('sector').apply(
            lambda x: (x['profit_ratio'] > 0).mean()
        ).round(3)
        st.bar_chart(wins_by_sector)

        # Recent Trades
        st.header('Recent Trades')
        recent_trades = trades_df.tail(10)[
            ['pair', 'profit_ratio', 'trade_duration', 'sector']
        ]
        st.dataframe(recent_trades)

if __name__ == '__main__':
    dashboard = FreqtradeDashboard()
    dashboard.create_dashboard()
