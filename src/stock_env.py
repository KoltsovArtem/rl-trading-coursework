import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta
import torch
import random

class StockEnv(gym.Env):
    """
    Окружение для торговли одной акцией (например, AAPL).
    Обсервация (state): [price_norm, sma20_norm, rsi14_norm, balance_norm, shares_norm]
    Действия (actions): Discrete(3) = {0=Buy10%, 1=Hold, 2=Sell10%}
    Reward = процентное изменение net_worth (шаговая прибыль в %)
    Есть комиссия (0.5%) и ограничение объёма покупки (5% баланса за сделку).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame = None, csv_path: str = None,
                 start_date: str = None, end_date: str = None,
                 initial_balance: float = 100_000):

        super(StockEnv, self).__init__()

        # ---------- 1) Загрузка и фильтрация данных ----------
        if df is None:
            if csv_path is None:
                raise ValueError("Нужно передать либо df, либо csv_path.")
            raw = pd.read_csv(csv_path, parse_dates=True, index_col=0)
            raw.index = pd.to_datetime(raw.index)
            if start_date is not None:
                raw = raw.loc[raw.index >= pd.to_datetime(start_date)]
            if end_date is not None:
                raw = raw.loc[raw.index <= pd.to_datetime(end_date)]
            df = raw.copy().reset_index(drop=True)
        else:
            df = df.copy().reset_index(drop=True)

        # ---------- 2) Вычисляем техиндикаторы SMA20 и RSI14 ----------
        df["SMA20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df = df.dropna(subset=["Close", "SMA20", "RSI14"]).reset_index(drop=True)

        # ---------- 3) Сохраняем важные массивы ----------
        self.prices = df["Close"].values
        self.sma20   = df["SMA20"].values
        self.rsi14   = df["RSI14"].values

        self.n_steps = len(self.prices)
        self.current_step = 0

        # ---------- 4) Действия и наблюдения ----------
        self.action_space = spaces.Discrete(3)  # {0=Buy10%, 1=Hold, 2=Sell10%}
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # ---------- 5) Торговый счёт и параметры ----------
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.prev_net_worth = initial_balance

        self.commission = 0.005              # 0.5% комиссия за сделку
        self.max_investment_fraction = 0.05  # 5% баланса за одну покупку

        # Сида для среды (для воспроизводимости)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.action_space.seed(0)
        self.observation_space.seed(0)

    def _next_observation(self):
        """
        Возвращает текущее состояние:
        [price_norm, sma20_norm, rsi14_norm, balance_norm, shares_norm]
        """
        price = self.prices[self.current_step]
        price_norm = (price - self.prices.min()) / (self.prices.max() - self.prices.min())

        sma_val = self.sma20[self.current_step]
        sma_norm = (sma_val - self.sma20.min()) / (self.sma20.max() - self.sma20.min())

        rsi_val = self.rsi14[self.current_step]
        rsi_norm = rsi_val / 100.0

        balance_norm = self.balance / self.initial_balance

        max_possible_shares = (self.initial_balance * self.max_investment_fraction) / max(price, 1e-8)
        if max_possible_shares > 0:
            shares_norm = self.shares_held / max_possible_shares
        else:
            shares_norm = 0.0

        return np.array([price_norm, sma_norm, rsi_norm, balance_norm, shares_norm], dtype=np.float32)

    def step(self, action: int):
        """
        Делаем один шаг в среде:
        - action=0: Buy 10% от текущего баланса
        - action=1: Hold (ничего не делаем)
        - action=2: Sell 10% от текущей позиции
        Возвращает: obs, reward, done, info
        """

        done = False
        price = self.prices[self.current_step]
        prev_net = self.prev_net_worth
        commission_amt = price * self.commission

        # --- Sell 10% от позиции ---
        if action == 2 and self.shares_held > 0:
            sell_amount = max(1, int(0.10 * self.shares_held))
            proceeds = sell_amount * price - sell_amount * commission_amt
            self.shares_held -= sell_amount
            self.balance += proceeds

        # --- Buy 10% от баланса ---
        elif action == 0:
            max_spend = self.balance * self.max_investment_fraction
            num_to_buy = int(max_spend / (price * (1 + self.commission)))
            if num_to_buy > 0:
                cost = num_to_buy * price + num_to_buy * commission_amt
                if self.balance >= cost:
                    self.shares_held += num_to_buy
                    self.balance -= cost

        # --- Hold (action == 1): ничего не делаем ---

        # Переходим к следующему дню/шагу
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        # Считаем новое net_worth
        new_price = self.prices[self.current_step]
        self.net_worth = self.balance + self.shares_held * new_price

        # Шаговая награда — процентная дельта капитала
        reward = (self.net_worth - prev_net) / max(prev_net, 1e-8)
        self.prev_net_worth = self.net_worth

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        """
        Сбрасывает среду в «нулевое» состояние (начало периода):
        balance = initial_balance, shares_held = 0, current_step = 0
        """
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation()

    def render(self, mode='human'):
        """
        Печатает текущее состояние (debug-поле)
        """
        print(
            f"Step: {self.current_step}, "
            f"Balance: {self.balance:.2f}, "
            f"Shares: {self.shares_held}, "
            f"Net worth: {self.net_worth:.2f}"
        )
