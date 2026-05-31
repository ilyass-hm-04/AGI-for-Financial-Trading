# -*- coding: utf-8 -*-
"""
train.py

Main training script for the AGI Financial Trading agent.

Design goal: ONE PPO agent that generalises to any stock without
per-stock hyperparameter tuning.

Key tricks
----------
1. Global StandardScaler (fitted across ALL training stocks in data.py)
   → every stock's features live in the same normalised space.
2. Multi-asset TradingEnv (random stock at each reset)
   → implicit domain randomisation / curriculum.
3. PPO with a deeper MLP (256-256-128) to handle the 204-dim observation.
4. Evaluation on held-out test stocks (NVDA, META) to verify generalisation.

Usage
-----
Quick smoke-test (50k steps, CPU-only):
    python train.py --timesteps 50000

Full training run:
    python train.py

Evaluate afterwards:
    python evaluate.py --model-path models/best_model --ticker NVDA
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from data import prepare_all, Train_data, Test_data
from trading_environment import TradingEnv
from callbacks import PortfolioMetricsCallback


# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # --- Data ---
    train_tickers = Train_data,          # ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    test_tickers  = Test_data,           # ["NVDA", "META"]
    start_date    = "2015-01-01",
    end_date      = "2023-12-31",

    # --- Environment ---
    window_size      = 20,
    initial_balance  = 10_000.0,
    transaction_cost = 0.001,
    reward_window    = 50,
    # Reward weights
    w_return      = 1.0,
    w_volatility  = 0.1,
    w_drawdown    = 0.5,
    w_transaction = 0.1,

    # --- PPO ---
    timesteps        = 2_000_000,
    n_envs           = 4,          # parallel envs (set to 1 for debugging)
    learning_rate    = 3e-4,
    n_steps          = 2048,       # steps per env before each PPO update
    batch_size       = 256,
    n_epochs         = 10,
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_range       = 0.2,
    ent_coef         = 0.05,
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
    # Network
    net_arch         = [256, 256, 128],

    # --- Callbacks ---
    eval_freq        = 50_000,     # evaluate every N training steps
    n_eval_episodes  = 10,

    # --- Persistence ---
    model_dir        = "models",
    log_dir          = "logs",
    scaler_path      = "models/scaler.pkl",
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(args) -> tuple[dict, dict, object]:
    """
    Download and normalise train + test data using a SHARED global scaler.

    The scaler is fitted ONLY on training stocks, then applied to test stocks
    as well — exactly as you would do in a real deployment.

    Returns
    -------
    train_data : dict[str, pd.DataFrame]   normalised + Close
    test_data  : dict[str, pd.DataFrame]   normalised + Close (same scaler)
    scaler     : StandardScaler
    """
    print("=" * 60)
    print("  Downloading & processing training data...")
    print(f"  Tickers : {args.train_tickers}")
    print(f"  Period  : {args.start_date} -> {args.end_date}")
    print("=" * 60)

    train_data, scaler = prepare_all(
        tickers      = args.train_tickers,
        start        = args.start_date,
        end          = args.end_date,
        fit_scaler   = True,
    )

    print(f"\n  Downloading & processing test data...")
    print(f"  Tickers : {args.test_tickers}")

    test_data, _ = prepare_all(
        tickers      = args.test_tickers,
        start        = args.start_date,
        end          = args.end_date,
        scaler       = scaler,
        fit_scaler   = False,   # ← re-use train scaler, never refit
    )

    print(f"\n  [OK] Train stocks : {list(train_data.keys())}")
    print(f"  [OK] Test  stocks : {list(test_data.keys())}")
    for t, df in train_data.items():
        print(f"    {t:6s} -> {len(df):,} rows")

    return train_data, test_data, scaler


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _make_single_env(stock_data: dict, args, seed: int = 0):
    """Returns a callable that creates and wraps a TradingEnv."""
    def _init():
        env = TradingEnv(
            df               = stock_data,
            window_size      = args.window_size,
            initial_balance  = args.initial_balance,
            transaction_cost = args.transaction_cost,
            reward_window    = args.reward_window,
            w_return         = args.w_return,
            w_volatility     = args.w_volatility,
            w_drawdown       = args.w_drawdown,
            w_transaction    = args.w_transaction,
        )
        env = Monitor(env)  # wraps for SB3 episode stats logging
        env.reset(seed=seed)
        return env
    return _init


def make_train_vec_env(train_data: dict, args):
    """
    Vectorised training environment: n_envs parallel copies of the
    multi-asset TradingEnv (each will randomly sample a different stock).
    """
    env_fns = [
        _make_single_env(train_data, args, seed=i)
        for i in range(args.n_envs)
    ]
    # DummyVecEnv is simpler and avoids pickle issues on Windows
    return DummyVecEnv(env_fns)


def make_eval_env(eval_data: dict, args):
    """
    Single (non-vectorised) environment for evaluation callbacks.
    Uses the test stocks so metrics reflect generalisation ability.
    """
    return TradingEnv(
        df               = eval_data,
        window_size      = args.window_size,
        initial_balance  = args.initial_balance,
        transaction_cost = args.transaction_cost,
        reward_window    = args.reward_window,
        w_return         = args.w_return,
        w_volatility     = args.w_volatility,
        w_drawdown       = args.w_drawdown,
        w_transaction    = args.w_transaction,
    )


# ---------------------------------------------------------------------------
# PPO Model
# ---------------------------------------------------------------------------

def build_model(vec_env, args) -> PPO:
    """
    Instantiate PPO with a custom MLP policy.

    The 256-256-128 network is deliberately deeper than SB3's default
    (64-64) to handle the 204-dim observation space comfortably.

    ent_coef=0.01 adds a small entropy bonus that helps the agent explore
    different actions (buy / hold / sell) during the early training phase.
    """
    policy_kwargs = dict(
        net_arch       = args.net_arch,
        activation_fn  = torch.nn.ReLU,
    )

    model = PPO(
        policy         = "MlpPolicy",
        env            = vec_env,
        learning_rate  = args.learning_rate,
        n_steps        = args.n_steps,
        batch_size     = args.batch_size,
        n_epochs       = args.n_epochs,
        gamma          = args.gamma,
        gae_lambda     = args.gae_lambda,
        clip_range     = args.clip_range,
        ent_coef       = args.ent_coef,
        vf_coef        = args.vf_coef,
        max_grad_norm  = args.max_grad_norm,
        target_kl      = 0.1,           # stop l'update si KL > 0.15 (1.5×0.1) → équilibre stabilité/vitesse
        policy_kwargs  = policy_kwargs,
        tensorboard_log = args.log_dir,
        verbose        = 1,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n  Policy architecture : {args.net_arch}")
    print(f"  Total parameters   : {total_params:,}")
    print(f"  Observation dim    : {vec_env.observation_space.shape[0]}")
    print(f"  Action dim         : {vec_env.action_space.n}")

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Full training pipeline."""

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)

    # 1. Data
    train_data, test_data, scaler = prepare_data(args)

    # Save scaler so evaluate.py can re-use it without re-downloading data
    scaler_dir = os.path.dirname(args.scaler_path)
    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
    with open(args.scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n  [OK] Scaler saved -> {args.scaler_path}")

    # 2. Environments
    print("\n  Building environments...")
    train_vec_env = make_train_vec_env(train_data, args)
    eval_env      = make_eval_env(test_data, args)   # test stocks for eval

    # 3. Model
    print("\n  Building PPO model...")
    model = build_model(train_vec_env, args)

    # 4. Callbacks
    portfolio_cb = PortfolioMetricsCallback(
        eval_env              = eval_env,
        eval_freq             = max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes       = args.n_eval_episodes,
        best_model_save_path  = args.model_dir,
        verbose               = 1,
    )

    # 5. Train
    print(f"\n{'='*60}")
    print(f"  Starting training for {args.timesteps:,} timesteps …")
    print(f"  Parallel envs : {args.n_envs}")
    print(f"  TensorBoard   : tensorboard --logdir {args.log_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()
    model.learn(
        total_timesteps = args.timesteps,
        callback        = [portfolio_cb],
        tb_log_name     = "PPO_trading",
        reset_num_timesteps = True,
        progress_bar    = True,
    )
    elapsed = time.time() - t0

    # 6. Save final model
    final_path = os.path.join(args.model_dir, "final_model")
    model.save(final_path)

    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best  model -> {args.model_dir}/best_model.zip")
    print(f"  Final model -> {final_path}.zip")
    print(f"{'='*60}")

    # 7. Quick generalisation check on test stocks
    print("\n  Quick generalisation check on test stocks...\n")
    _quick_eval(model, test_data, args)

    train_vec_env.close()
    eval_env.close()

    return model, scaler


# ---------------------------------------------------------------------------
# Quick post-training evaluation
# ---------------------------------------------------------------------------

def _quick_eval(model, test_data: dict, args):
    """
    Run 1 deterministic episode per test ticker and print a short report.
    For a detailed report + equity curve, use evaluate.py.
    """
    for ticker, df in test_data.items():
        env = TradingEnv(
            df               = df,
            window_size      = args.window_size,
            initial_balance  = args.initial_balance,
            transaction_cost = args.transaction_cost,
            reward_window    = args.reward_window,
        )
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        net_worths   = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += float(reward)
            net_worths.append(info.get("net_worth", args.initial_balance))

        final_nw   = net_worths[-1]
        total_return = (final_nw / args.initial_balance - 1) * 100

        # Sharpe
        rewards_arr = np.diff(net_worths) / (np.array(net_worths[:-1]) + 1e-9)
        sharpe = (rewards_arr.mean() / (rewards_arr.std() + 1e-9)) * np.sqrt(252)

        # Max drawdown
        nw_arr      = np.array(net_worths)
        running_max = np.maximum.accumulate(nw_arr)
        max_dd      = float(((running_max - nw_arr) / (running_max + 1e-9)).max())

        print(
            f"  {ticker:6s} | Return: {total_return:+.1f}% | "
            f"Net worth: ${final_nw:,.2f} | "
            f"Sharpe: {sharpe:.3f} | "
            f"Max DD: {max_dd:.2%}"
        )
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a PPO agent on multiple stocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = DEFAULTS

    # Data
    p.add_argument("--train-tickers", nargs="+", default=d["train_tickers"],
                   metavar="TICKER", help="Training tickers.")
    p.add_argument("--test-tickers",  nargs="+", default=d["test_tickers"],
                   metavar="TICKER", help="Test tickers (held out).")
    p.add_argument("--start-date",    default=d["start_date"])
    p.add_argument("--end-date",      default=d["end_date"])

    # Environment
    p.add_argument("--window-size",      type=int,   default=d["window_size"])
    p.add_argument("--initial-balance",  type=float, default=d["initial_balance"])
    p.add_argument("--transaction-cost", type=float, default=d["transaction_cost"])
    p.add_argument("--reward-window",    type=int,   default=d["reward_window"])
    p.add_argument("--w-return",         type=float, default=d["w_return"])
    p.add_argument("--w-volatility",     type=float, default=d["w_volatility"])
    p.add_argument("--w-drawdown",       type=float, default=d["w_drawdown"])
    p.add_argument("--w-transaction",    type=float, default=d["w_transaction"])

    # PPO
    p.add_argument("--timesteps",     type=int,   default=d["timesteps"])
    p.add_argument("--n-envs",        type=int,   default=d["n_envs"])
    p.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    p.add_argument("--n-steps",       type=int,   default=d["n_steps"])
    p.add_argument("--batch-size",    type=int,   default=d["batch_size"])
    p.add_argument("--n-epochs",      type=int,   default=d["n_epochs"])
    p.add_argument("--gamma",         type=float, default=d["gamma"])
    p.add_argument("--gae-lambda",    type=float, default=d["gae_lambda"])
    p.add_argument("--clip-range",    type=float, default=d["clip_range"])
    p.add_argument("--ent-coef",      type=float, default=d["ent_coef"])
    p.add_argument("--vf-coef",       type=float, default=d["vf_coef"])
    p.add_argument("--max-grad-norm", type=float, default=d["max_grad_norm"])
    p.add_argument("--net-arch",      type=int,   nargs="+",
                   default=d["net_arch"], metavar="N",
                   help="Hidden layer sizes. E.g. --net-arch 256 256 128")


    # Callbacks
    p.add_argument("--eval-freq",       type=int, default=d["eval_freq"])
    p.add_argument("--n-eval-episodes", type=int, default=d["n_eval_episodes"])

    # Paths
    p.add_argument("--model-dir",   default=d["model_dir"])
    p.add_argument("--log-dir",     default=d["log_dir"])
    p.add_argument("--scaler-path", default=d["scaler_path"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
