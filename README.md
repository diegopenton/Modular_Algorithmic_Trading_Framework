# Modular Algorithmic Trading Framework (Currently in Hiatus for Mid-Term Exam season)

A Python-based **algorithmic trading system** built with a **tree-trunk & branches architecture**.  
Each branch is a model (RSI, GARCH, ML), and the trunk combines them into trading decisions with risk controls.

## Features
- Modular design with pluggable branch models  
- RSI DecisionTree branch with walk-forward validation  
- Backtesting engine (Sharpe, Max Drawdown)  
- Policy layer for signal filtering & position sizing  

## Quick Start
```bash
git clone https://github.com/<your-username>/tree-trunk-LSTM-GARCH.git
cd tree-trunk-LSTM-GARCH
pip install -r requirements.txt
python -m src.backtest.walkforward --ticker AAPL --start 2018-01-01 --end 2023-01-01
