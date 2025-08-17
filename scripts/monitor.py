import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from pathlib import Path

class TradingMonitor:
    def __init__(self):
        st.set_page_config(page_title='Trading Monitor', layout='wide')
        self.db_path = Path('user_data/trades')

    def load_trades(self):
        """Load current trades from freqtrade database"""
        try:
            with open(self.db_path / 'trades.json', 'r') as f:
                return pd.DataFrame(json.load(f))
        except:
            return pd.DataFrame()

    def create_dashboard(self):
        st.title('📈 Live Trading Monitor')

        # Auto-refresh every 60 seconds
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        trades_df = self.load_trades()
        
        with col1:
            st.metric(
                "Current Balance",
                f"${trades_df['profit_abs'].sum():.2f}",
                f"{trades_df['profit_ratio'].mean()*100:.1f}%"
            )
        
        with col2:
            st.metric(
                "Open Trades",
                len(trades_df[trades_df['is_open']]),
                "Max: 3"
            )
        
        with col3:
            drawdown = self.calculate_drawdown(trades_df)
            st.metric(
                "Max Drawdown",
                f"{drawdown:.1f}%",
                "Target: <20%"
            )
        
        with col4:
            win_rate = len(trades_df[trades_df['profit_ratio'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
            st.metric(
                "Win Rate",
                f"{win_rate*100:.1f}%",
                "Target: >60%"
            )

        # Active Trades Table
        st.subheader("🔄 Active Trades")
        active_trades = trades_df[trades_df['is_open']].copy()
        if not active_trades.empty:
            st.dataframe(
                active_trades[['pair', 'open_rate', 'current_rate', 'profit_ratio', 'open_date']]
            )

        # Recent Closed Trades
        st.subheader("✅ Recent Closed Trades")
        closed_trades = trades_df[~trades_df['is_open']].tail(10)
        if not closed_trades.empty:
            st.dataframe(
                closed_trades[['pair', 'close_rate', 'profit_ratio', 'close_date', 'trade_duration']]
            )

        # Profit Chart
        st.subheader("💰 Cumulative Profit")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trades_df['close_date'],
                y=trades_df['profit_abs'].cumsum(),
                mode='lines',
                name='Profit'
            )
        )
        st.plotly_chart(fig)

        # Auto-refresh
        time_since_refresh = datetime.now() - st.session_state.last_refresh
        if time_since_refresh > timedelta(seconds=60):
            st.session_state.last_refresh = datetime.now()
            st.experimental_rerun()

    @staticmethod
    def calculate_drawdown(trades_df):
        if trades_df.empty:
            return 0
        cumulative = trades_df['profit_abs'].cumsum()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max * 100
        return abs(drawdown.min())

if __name__ == "__main__":
    monitor = TradingMonitor()
    monitor.create_dashboard()
