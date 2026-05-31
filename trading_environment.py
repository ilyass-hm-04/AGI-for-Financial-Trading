# -*- coding: utf-8 -*-
"""
trading_environment.py

Classe unique TradingEnv — supporte un seul actif OU plusieurs marchés.

Usage mono-actif :
    env = TradingEnv(df=train_data["AAPL"])

Usage multi-actifs (un marché aléatoire est sélectionné à chaque reset) :
    env = TradingEnv(df=train_data)   # train_data = {"AAPL": df, "MSFT": df, ...}

Reward shaping :
    Gain brut − Volatilité − Drawdown − Pénalité de transaction
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Union


class TradingEnv(gym.Env):
    """
    Environnement Gymnasium pour le trading, mono ou multi-actifs.

    Paramètres
    ----------
    df : pd.DataFrame | dict[str, pd.DataFrame]
        - DataFrame unique  → mode mono-actif (toujours le même marché).
        - Dictionnaire      → mode multi-actifs (marché aléatoire à chaque reset).
        Chaque DataFrame doit contenir les colonnes produites par data.py :
        ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_width', 'ATR',
         'Return_1d', 'Return_5d', 'Return_20d', 'Volume_norm', 'Close']
    window_size : int
        Nombre de pas de temps dans la fenêtre d'observation.
    initial_balance : float
        Capital de départ (ex. 10 000 $).
    transaction_cost : float
        Taux de frais de transaction (0.001 = 0.1 %).
    reward_window : int
        Nombre d'étapes mémorisées pour calculer la volatilité roulante.
    w_return : float
        Poids du rendement brut dans la récompense.
    w_volatility : float
        Poids de la pénalité de volatilité (écart-type des rendements récents).
    w_drawdown : float
        Poids de la pénalité de drawdown (appliqué au carré pour amplifier les grosses chutes).
    w_transaction : float
        Poids de la pénalité de frais (limite l'overtrading).
    render_mode : str | None
        'human' pour afficher les étapes dans le terminal, None sinon.
    """

    metadata = {"render_modes": ["human"]}

    # Colonnes de marché normalisées produites par data.py
    MARKET_FEATURES = [
        "RSI", "MACD", "MACD_signal", "MACD_diff",
        "BB_width", "ATR",
        "Return_1d", "Return_5d", "Return_20d", "Volume_norm",
    ]
    N_FEATURES  = len(MARKET_FEATURES)   # 10
    N_PORTFOLIO = 4                       # position | pnl_non_réalisé | durée | capital_ratio

    def __init__(
        self,
        df: Union[pd.DataFrame, dict],
        window_size: int = 20,
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.001,
        reward_window: int = 50,
        # --- Poids du reward shaping ---
        w_return: float = 1.0,
        w_volatility: float = 0.5,
        w_drawdown: float = 2.0,
        w_transaction: float = 0.1,
        render_mode=None,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # MODE MONO ou MULTI-ACTIFS
        # ------------------------------------------------------------------
        if isinstance(df, dict):
            # Mode multi-actifs : stocke le dictionnaire et extrait les tickers
            if len(df) == 0:
                raise ValueError("Le dictionnaire de DataFrames est vide.")
            self._stock_data = {k: v.reset_index(drop=True) for k, v in df.items()}
            self._tickers     = list(df.keys())
            self._multi_mode  = True
        elif isinstance(df, pd.DataFrame):
            # Mode mono-actif : encapsule dans un dict avec un ticker générique
            self._stock_data = {"MARKET": df.reset_index(drop=True)}
            self._tickers     = ["MARKET"]
            self._multi_mode  = False
        else:
            raise TypeError(
                f"'df' doit être un pd.DataFrame ou un dict[str, pd.DataFrame], "
                f"reçu : {type(df)}"
            )

        # Valider les colonnes de chaque DataFrame
        required_cols = self.MARKET_FEATURES + ["Close"]
        for ticker, frame in self._stock_data.items():
            missing = [c for c in required_cols if c not in frame.columns]
            if missing:
                raise ValueError(
                    f"[{ticker}] Colonnes manquantes : {missing}\n"
                    f"Colonnes disponibles : {list(frame.columns)}"
                )

        # ------------------------------------------------------------------
        # HYPERPARAMÈTRES
        # ------------------------------------------------------------------
        self.window_size      = window_size
        self.initial_balance  = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_window    = reward_window
        self.render_mode      = render_mode

        # Poids du reward shaping
        self.w_return      = w_return
        self.w_volatility  = w_volatility
        self.w_drawdown    = w_drawdown
        self.w_transaction = w_transaction

        obs_dim = self.window_size * self.N_FEATURES + self.N_PORTFOLIO
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)

        self.returns_history = deque(maxlen=reward_window)
        self.current_ticker  = self._tickers[0]
        self.df              = self._stock_data[self.current_ticker]

        self.t             = window_size
        self.position      = 0
        self.capital       = float(initial_balance)
        self.entry_price   = 0.0
        self.shares_held   = 0.0
        self.step_count    = 0
        self.peak_portfolio = float(initial_balance)


    @property
    def tickers(self) -> list:
        """Liste des tickers disponibles."""
        return self._tickers

    @property
    def is_multi(self) -> bool:
        """True si l'environnement est en mode multi-actifs."""
        return self._multi_mode

    def _portfolio_value(self) -> float:
        """Valeur totale : cash + valeur des actions détenues au prix courant."""
        price = self.df["Close"].iloc[self.t]
        return self.capital + self.shares_held * price

    def _get_obs(self) -> np.ndarray:
        """
        Construit le vecteur d'observation :
          [fenêtre_marché (window_size x N_FEATURES) | état_portefeuille (N_PORTFOLIO)]
        """
        # Fenêtre de marché normalisée → shape (window_size × N_FEATURES,)
        window_data = self.df.iloc[self.t - self.window_size: self.t]
        market_obs  = window_data[self.MARKET_FEATURES].values.flatten().astype(np.float32)

        # PnL non réalisé en % (0 si pas en position)
        unrealized_pnl = 0.0
        if self.position == 1 and self.entry_price > 0:
            current_price  = self.df["Close"].iloc[self.t]
            unrealized_pnl = (current_price / self.entry_price) - 1.0

        # Capital normalisé par le solde initial
        capital_ratio = self.capital / (self.initial_balance + 1e-9)

        portfolio_obs = np.array([
            float(self.position),       # 0 = neutre, 1 = long
            float(unrealized_pnl),      # PnL latent en %
            self.step_count / 252.0,    # Durée normalisée (~1 an de bourse)
            float(capital_ratio),       # Richesse relative
        ], dtype=np.float32)

        return np.concatenate([market_obs, portfolio_obs])

    def _select_market(self, rng: np.random.Generator) -> None:
        """
        En mode multi-actifs : tire un ticker au hasard et charge son DataFrame.
        En mode mono-actif   : ne fait rien (le ticker reste le même).
        """
        if self._multi_mode:
            self.current_ticker = rng.choice(self._tickers)
        self.df = self._stock_data[self.current_ticker]


    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement.
        En mode multi-actifs, sélectionne un marché aléatoire.
        """
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Choisir le marché (aléatoire si multi-actifs)
        self._select_market(rng)

        # Réinitialiser les états
        self.t              = self.window_size
        self.position       = 0
        self.capital        = float(self.initial_balance)
        self.entry_price    = 0.0
        self.shares_held    = 0.0
        self.step_count     = 0
        self.peak_portfolio = float(self.initial_balance)
        self.returns_history.clear()

        info = {"ticker": self.current_ticker} if self._multi_mode else {}
        return self._get_obs(), info

    def step(self, action: int):
        """
        Exécute une action et retourne (observation, reward, terminated, truncated, info).

        Actions
        -------
        0 : Hold  — ne rien faire
        1 : Buy   — investir tout le capital disponible
        2 : Sell  — liquider toutes les actions
        """
        prev_portfolio = self._portfolio_value()
        old_position   = self.position
        current_price  = self.df["Close"].iloc[self.t]

        if action == 1 and self.position == 0:
            # BUY : investir tout le capital disponible
            cost_per_share = current_price * (1.0 + self.transaction_cost)
            if self.capital >= cost_per_share:
                self.shares_held = self.capital / cost_per_share
                self.capital     = 0.0
                self.entry_price = current_price
                self.position    = 1

        elif action == 2 and self.position == 1:
            # SELL : liquider toutes les actions et réinjecter le PnL réalisé
            revenue          = self.shares_held * current_price * (1.0 - self.transaction_cost)
            self.capital    += revenue   # ← PnL réalisé intégré dans le capital
            self.shares_held = 0.0
            self.entry_price = 0.0
            self.position    = 0

        self.t          += 1
        self.step_count += 1

        curr_portfolio = self._portfolio_value()
        step_return = (curr_portfolio - prev_portfolio) / (prev_portfolio + 1e-9)
        self.returns_history.append(step_return)

        sigma = float(np.std(self.returns_history)) if len(self.returns_history) >= 2 else 0.0

        if curr_portfolio > self.peak_portfolio:
            self.peak_portfolio = curr_portfolio
        drawdown = (self.peak_portfolio - curr_portfolio) / (self.peak_portfolio + 1e-9)
        tx_penalty = self.transaction_cost if (action != 0 and action != old_position) else 0.0

        # Récompense combinée : Gain − Volatilité − Drawdown − Frais
        reward = (
              self.w_return      * step_return
            - self.w_volatility  * sigma
            - self.w_drawdown    * drawdown
            - self.w_transaction * tx_penalty
        )

        terminated = self.t >= len(self.df) - 1
        truncated  = False

        info = {
            "ticker":      self.current_ticker,
            "net_worth":   curr_portfolio,
            "capital":     self.capital,
            "shares_held": self.shares_held,
            "step_return": step_return,
            "volatility":  sigma,
            "drawdown":    drawdown,
            "position":    self.position,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            price     = self.df["Close"].iloc[self.t - 1]
            net_worth = self._portfolio_value()
            pos       = "LONG" if self.position == 1 else "NEUTRE"
            ticker    = f"[{self.current_ticker}] " if self._multi_mode else ""
            print(
                f"  {ticker}Étape {self.step_count:4d} | "
                f"Prix {price:8.2f} $ | "
                f"Valeur nette {net_worth:10.2f} $ | "
                f"Capital {self.capital:10.2f} $ | "
                f"Position: {pos}"
            )
