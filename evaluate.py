# -*- coding: utf-8 -*-
"""
evaluate.py

Standalone evaluation script for the trained AGI Financial Trading agent.

Loads a saved PPO model and runs it on any ticker (train or test),
then produces:
  - A full performance report printed to console
  - An equity curve plot (PNG) saved to disk
  - A CSV of step-by-step trades and portfolio values

Usage
-----
# Evaluate the best model on a held-out stock:
    python evaluate.py --model-path models/best_model --ticker NVDA

# Evaluate on a custom ticker not in the original dataset:
    python evaluate.py --model-path models/best_model --ticker PLTR --start-date 2020-01-01

# Compare agent vs buy-and-hold for multiple tickers:
    python evaluate.py --model-path models/best_model --ticker NVDA META AAPL
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering → saves PNG without display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from stable_baselines3 import PPO

from data import prepare_all
from trading_environment import TradingEnv


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

def evaluate_ticker(
    model,
    df: pd.DataFrame,
    ticker: str,
    initial_balance: float = 10_000.0,
    window_size: int = 20,
    transaction_cost: float = 0.001,
    reward_window: int = 50,
    deterministic: bool = True,
) -> pd.DataFrame:
    """
    Run one full episode and return a step-level DataFrame with columns:
        step, price, action, position, capital, shares_held,
        net_worth, step_return, drawdown, reward
    """
    env = TradingEnv(
        df               = df,
        window_size      = window_size,
        initial_balance  = initial_balance,
        transaction_cost = transaction_cost,
        reward_window    = reward_window,
    )

    obs, _ = env.reset(seed=42)
    done   = False
    rows   = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        rows.append({
            "step":       env.step_count,
            "price":      df["Close"].iloc[env.t - 1],
            "action":     ["Hold", "Buy", "Sell"][int(action)],
            "position":   info["position"],
            "capital":    info["capital"],
            "shares_held": info["shares_held"],
            "net_worth":  info["net_worth"],
            "step_return": info["step_return"],
            "drawdown":   info["drawdown"],
            "reward":     float(reward),
        })

    env.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(episode_df: pd.DataFrame, initial_balance: float) -> dict:
    """Compute a rich set of performance metrics from an episode DataFrame."""
    nw = episode_df["net_worth"].values

    total_return  = (nw[-1] / initial_balance - 1) * 100
    step_rets     = np.diff(nw) / (nw[:-1] + 1e-9)
    sharpe        = (step_rets.mean() / (step_rets.std() + 1e-9)) * np.sqrt(252)

    running_max   = np.maximum.accumulate(nw)
    drawdowns     = (running_max - nw) / (running_max + 1e-9)
    max_drawdown  = float(drawdowns.max())
    calmar        = total_return / (max_drawdown * 100 + 1e-9)

    n_buy         = (episode_df["action"] == "Buy").sum()
    n_sell        = (episode_df["action"] == "Sell").sum()
    n_hold        = (episode_df["action"] == "Hold").sum()

    # Win rate: fraction of completed long trades that were profitable
    profits = []
    buys    = episode_df[episode_df["action"] == "Buy"].index.tolist()
    sells   = episode_df[episode_df["action"] == "Sell"].index.tolist()
    for b, s in zip(buys, sells[:len(buys)]):
        bp = episode_df.loc[b, "price"]
        sp = episode_df.loc[s, "price"]
        profits.append(sp > bp)
    win_rate = np.mean(profits) * 100 if profits else 0.0

    return {
        "total_return_%":  round(total_return, 2),
        "final_net_worth": round(nw[-1], 2),
        "sharpe_ratio":    round(float(sharpe), 4),
        "max_drawdown_%":  round(max_drawdown * 100, 2),
        "calmar_ratio":    round(calmar, 4),
        "n_buy":           int(n_buy),
        "n_sell":          int(n_sell),
        "n_hold":          int(n_hold),
        "win_rate_%":      round(win_rate, 1),
        "total_steps":     len(episode_df),
    }


# ---------------------------------------------------------------------------
# Buy-and-hold baseline
# ---------------------------------------------------------------------------

def buy_and_hold(df: pd.DataFrame, initial_balance: float = 10_000.0) -> np.ndarray:
    """
    Simulate a naive buy-and-hold strategy: buy on day 1, hold forever.
    Returns array of net worth at each step (same length as the agent episode).
    """
    prices     = df["Close"].values
    cost       = prices[0] * 1.001          # 0.1% transaction cost
    shares     = initial_balance / cost
    net_worths = shares * prices
    return net_worths


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    episode_df:  pd.DataFrame,
    bah_nw:      np.ndarray,
    ticker:      str,
    metrics:     dict,
    initial_balance: float,
    output_dir:  str = "results",
) -> str:
    """
    Save a 3-panel figure:
      - Panel 1: Equity curve (agent vs buy-and-hold)
      - Panel 2: Price + buy/sell markers
      - Panel 3: Drawdown over time
    """
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0d1117")
    gs  = GridSpec(3, 1, figure=fig, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    steps = episode_df["step"].values
    nw    = episode_df["net_worth"].values
    price = episode_df["price"].values

    # ── Panel 1: Equity curves ─────────────────────────────────────────────
    ax1.set_facecolor("#161b22")
    bah_aligned = bah_nw[:len(nw)] if len(bah_nw) >= len(nw) else np.pad(
        bah_nw, (0, len(nw) - len(bah_nw)), mode="edge"
    )

    ax1.plot(steps, nw,          color="#58a6ff", lw=2.0,  label="PPO Agent")
    ax1.plot(steps, bah_aligned, color="#f0883e", lw=1.5,  label="Buy & Hold",
             linestyle="--", alpha=0.85)
    ax1.axhline(initial_balance, color="#8b949e", lw=1, linestyle=":")
    ax1.fill_between(steps, initial_balance, nw,
                     where=(nw >= initial_balance),
                     alpha=0.15, color="#58a6ff", interpolate=True)
    ax1.fill_between(steps, initial_balance, nw,
                     where=(nw < initial_balance),
                     alpha=0.15, color="#f85149", interpolate=True)

    title_str = (
        f"{ticker}  |  Return: {metrics['total_return_%']:+.1f}%  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.3f}  |  "
        f"Max DD: {metrics['max_drawdown_%']:.1f}%"
    )
    ax1.set_title(title_str, color="#e6edf3", fontsize=13, pad=10)
    ax1.set_ylabel("Portfolio Value ($)", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    ax1.legend(loc="upper left", framealpha=0.3, labelcolor="#e6edf3")
    for spine in ax1.spines.values():
        spine.set_color("#30363d")

    # ── Panel 2: Price + trade signals ────────────────────────────────────
    ax2.set_facecolor("#161b22")
    ax2.plot(steps, price, color="#c9d1d9", lw=1.2, label="Close Price")

    buy_steps  = episode_df[episode_df["action"] == "Buy"]["step"].values
    sell_steps = episode_df[episode_df["action"] == "Sell"]["step"].values
    buy_prices  = episode_df[episode_df["action"] == "Buy"]["price"].values
    sell_prices = episode_df[episode_df["action"] == "Sell"]["price"].values

    ax2.scatter(buy_steps,  buy_prices,  marker="^", color="#3fb950", s=60,
                zorder=5, label=f"Buy  ({metrics['n_buy']})")
    ax2.scatter(sell_steps, sell_prices, marker="v", color="#f85149", s=60,
                zorder=5, label=f"Sell ({metrics['n_sell']})")

    ax2.set_ylabel("Price ($)", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    ax2.legend(loc="upper left", framealpha=0.3, labelcolor="#e6edf3")
    for spine in ax2.spines.values():
        spine.set_color("#30363d")

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────
    ax3.set_facecolor("#161b22")
    dd = episode_df["drawdown"].values * 100
    ax3.fill_between(steps, 0, -dd, color="#f85149", alpha=0.6)
    ax3.plot(steps, -dd, color="#f85149", lw=1.0)
    ax3.set_ylabel("Drawdown (%)", color="#8b949e")
    ax3.set_xlabel("Step",          color="#8b949e")
    ax3.tick_params(colors="#8b949e")
    for spine in ax3.spines.values():
        spine.set_color("#30363d")

    plt.suptitle(
        f"AGI Financial Trading — {ticker}",
        color="#e6edf3", fontsize=15, y=0.98,
    )

    path = os.path.join(output_dir, f"{ticker}_equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Plot saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(ticker: str, metrics: dict):
    box = "-" * 48
    print(f"\n  +{box}+")
    print(f"  |{'  Performance Report: ' + ticker:^48}|")
    print(f"  +{box}+")
    rows = [
        ("Total Return",  f"{metrics['total_return_%']:+.2f} %"),
        ("Final Net Worth", f"${metrics['final_net_worth']:,.2f}"),
        ("Sharpe Ratio",  f"{metrics['sharpe_ratio']:.4f}"),
        ("Max Drawdown",  f"{metrics['max_drawdown_%']:.2f} %"),
        ("Calmar Ratio",  f"{metrics['calmar_ratio']:.4f}"),
        ("Win Rate",      f"{metrics['win_rate_%']:.1f} %"),
        ("Trades: Buy / Sell / Hold",
         f"{metrics['n_buy']} / {metrics['n_sell']} / {metrics['n_hold']}"),
        ("Total Steps",   f"{metrics['total_steps']:,}"),
    ]
    for label, value in rows:
        print(f"  |  {label:<28}{value:>16}  |")
    print(f"  +{box}+\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    model_path = args.model_path
    if not model_path.endswith(".zip"):
        model_path = model_path + ".zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run `python train.py` first to train the agent."
        )
    print(f"\n  Loading model: {model_path}")
    model = PPO.load(model_path)

    # ── Load scaler ───────────────────────────────────────────────────────
    if not os.path.exists(args.scaler_path):
        raise FileNotFoundError(
            f"Scaler not found: {args.scaler_path}\n"
            "Run `python train.py` first — the scaler is saved automatically."
        )
    with open(args.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Scaler loaded : {args.scaler_path}")

    # ── Fetch & process data for each requested ticker ────────────────────
    print(f"\n  Fetching data for: {args.tickers}")
    eval_data, _ = prepare_all(
        tickers      = args.tickers,
        start        = args.start_date,
        end          = args.end_date,
        scaler       = scaler,
        fit_scaler   = False,   # always use the training scaler
    )

    # ── Evaluate each ticker ──────────────────────────────────────────────
    all_metrics = {}
    for ticker, df in eval_data.items():
        print(f"\n{'='*55}")
        print(f"  Evaluating: {ticker}")

        episode_df = evaluate_ticker(
            model            = model,
            df               = df,
            ticker           = ticker,
            initial_balance  = args.initial_balance,
            window_size      = args.window_size,
            transaction_cost = args.transaction_cost,
            reward_window    = args.reward_window,
            deterministic    = True,
        )

        metrics = compute_metrics(episode_df, args.initial_balance)
        all_metrics[ticker] = metrics
        print_report(ticker, metrics)

        # Buy-and-hold baseline
        bah = buy_and_hold(df, args.initial_balance)

        # Plot
        plot_results(
            episode_df      = episode_df,
            bah_nw          = bah,
            ticker          = ticker,
            metrics         = metrics,
            initial_balance = args.initial_balance,
            output_dir      = args.output_dir,
        )

        # Save step-level CSV
        csv_path = os.path.join(args.output_dir, f"{ticker}_trades.csv")
        episode_df.to_csv(csv_path, index=False)
        print(f"  [OK] Trades CSV  -> {csv_path}")

    # -- Summary table -----------------------------------------------------
    if len(all_metrics) > 1:
        print("\n  +-- Summary --------------------------------------------------+")
        header = f"  | {'Ticker':<8} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>9} |"
        print(header)
        print("  +------------------------------------------------------------+")
        for tk, m in all_metrics.items():
            row = (
                f"  | {tk:<8} "
                f"{m['total_return_%']:>+8.1f}% "
                f"{m['sharpe_ratio']:>8.3f} "
                f"{m['max_drawdown_%']:>7.1f}% "
                f"{m['win_rate_%']:>8.1f}%  |"
            )
            print(row)
        print("  +------------------------------------------------------------+\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained PPO trading agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path",  default="models/best_model",
                   help="Path to saved model (with or without .zip)")
    p.add_argument("--scaler-path", default="models/scaler.pkl",
                   help="Path to the fitted StandardScaler pickle")
    p.add_argument("--ticker",      nargs="+", dest="tickers",
                   default=["NVDA", "META"],
                   metavar="TICKER", help="Ticker(s) to evaluate")
    p.add_argument("--start-date",  default="2015-01-01")
    p.add_argument("--end-date",    default="2023-12-31")
    p.add_argument("--initial-balance",  type=float, default=10_000.0)
    p.add_argument("--window-size",      type=int,   default=20)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--reward-window",    type=int,   default=50)
    p.add_argument("--output-dir", default="results",
                   help="Directory to save plots and CSVs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
