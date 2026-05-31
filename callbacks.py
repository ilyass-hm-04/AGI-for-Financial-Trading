# -*- coding: utf-8 -*-
"""
callbacks.py

Custom Stable-Baselines3 callbacks for the AGI Financial Trading agent.

Callbacks
---------
PortfolioMetricsCallback
    Runs full evaluation episodes on a dedicated eval environment after every
    ``eval_freq`` training steps, then logs rich portfolio metrics to
    TensorBoard:
        - mean / std episode reward
        - mean final net worth
        - mean Sharpe ratio
        - mean maximum drawdown
    Also saves the best model (by mean reward) to disk.

CurriculumCallback  [stub]
    Placeholder for future curriculum learning (e.g. gradually increasing
    the window_size or the number of assets during training).
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


# ---------------------------------------------------------------------------
# Helper: compute per-episode metrics from a full rollout
# ---------------------------------------------------------------------------

def _run_episode(env, model) -> dict:
    """
    Roll out ONE episode using the given model.
    Returns a dict with:
        total_reward, final_net_worth, max_drawdown, sharpe_ratio
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    rewards = []
    net_worths = []
    peak = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += float(reward)
        rewards.append(float(reward))

        nw = info.get("net_worth", 0.0)
        net_worths.append(nw)
        if nw > peak:
            peak = nw

    # Sharpe ratio (annualized, from net worth returns — not from penalised rewards)
    nw_series = np.array(net_worths)
    if len(nw_series) > 1:
        daily_rets = np.diff(nw_series) / (nw_series[:-1] + 1e-9)
        sharpe = (daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Maximum drawdown
    nw_arr = np.array(net_worths)
    if len(nw_arr) > 0:
        running_max = np.maximum.accumulate(nw_arr)
        drawdowns = (running_max - nw_arr) / (running_max + 1e-9)
        max_dd = float(drawdowns.max())
    else:
        max_dd = 0.0

    return {
        "total_reward":    total_reward,
        "final_net_worth": net_worths[-1] if net_worths else 0.0,
        "sharpe_ratio":    float(sharpe),
        "max_drawdown":    max_dd,
    }


# ---------------------------------------------------------------------------
# PortfolioMetricsCallback
# ---------------------------------------------------------------------------

class PortfolioMetricsCallback(BaseCallback):
    """
    Evaluate the policy every ``eval_freq`` steps and log portfolio metrics
    to TensorBoard.

    Parameters
    ----------
    eval_env : gym.Env
        A non-vectorised TradingEnv used exclusively for evaluation.
        Can be in multi-asset mode so each evaluation episode may hit a
        different stock — giving a more honest generalisation estimate.
    eval_freq : int
        Number of *training* steps between evaluations.
    n_eval_episodes : int
        Number of episodes to average metrics over.
    best_model_save_path : str
        Directory where the best model (by mean reward) is saved.
    verbose : int
        0 = silent, 1 = print eval results.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 10,
        best_model_save_path: str = "models",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env            = eval_env
        self.eval_freq           = eval_freq
        self.n_eval_episodes     = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward    = -np.inf

        os.makedirs(best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True  # keep training

        # ---- Run evaluation episodes ----
        episode_metrics = [
            _run_episode(self.eval_env, self.model)
            for _ in range(self.n_eval_episodes)
        ]

        mean_reward    = np.mean([m["total_reward"]    for m in episode_metrics])
        std_reward     = np.std( [m["total_reward"]    for m in episode_metrics])
        mean_net_worth = np.mean([m["final_net_worth"] for m in episode_metrics])
        mean_sharpe    = np.mean([m["sharpe_ratio"]    for m in episode_metrics])
        mean_drawdown  = np.mean([m["max_drawdown"]    for m in episode_metrics])

        # ---- Log to TensorBoard ----
        self.logger.record("eval/mean_reward",     mean_reward)
        self.logger.record("eval/std_reward",      std_reward)
        self.logger.record("eval/mean_net_worth",  mean_net_worth)
        self.logger.record("eval/mean_sharpe",     mean_sharpe)
        self.logger.record("eval/mean_max_drawdown", mean_drawdown)
        self.logger.dump(self.num_timesteps)

        # ---- Console output ----
        if self.verbose >= 1:
            print(
                f"\n[Eval @ {self.num_timesteps:,} steps] "
                f"Reward: {mean_reward:.4f} +/- {std_reward:.4f} | "
                f"Net worth: ${mean_net_worth:,.2f} | "
                f"Sharpe: {mean_sharpe:.3f} | "
                f"Max DD: {mean_drawdown:.2%}"
            )

        # ---- Save best model ----
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            save_path = os.path.join(self.best_model_save_path, "best_model")
            self.model.save(save_path)
            if self.verbose >= 1:
                print(f"  [BEST] New best model saved -> {save_path}.zip")

        return True  # always continue training


# ---------------------------------------------------------------------------
# CurriculumCallback  (stub — extend for Phase 2)
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    [STUB] Curriculum learning callback.

    In a future phase this callback can:
    - Start training on easier markets (low-volatility stocks) and gradually
      introduce more volatile ones.
    - Increase window_size progressively as the agent matures.
    - Adjust reward weights (e.g. reduce w_volatility once the agent shows
      consistent positive returns).

    Currently it is a no-op — extend ``_on_step`` to implement curriculum
    logic.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # TODO: implement curriculum schedule
        return True
