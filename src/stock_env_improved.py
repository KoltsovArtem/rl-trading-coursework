import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta

class StockEnvImproved(gym.Env):
    """
    Улучшенная среда для PPO:
      - State: [price_norm, sma20_norm, rsi14_norm, balance_norm, shares_norm]
      - Action: Discrete(5):
          0 = Buy 20% от текущего баланса
          1 = Buy 50% от текущего баланса
          2 = Hold
          3 = Sell 20% от текущей позиции
          4 = Sell 50% от текущей позиции
      - Reward: процентное изменение net_worth относительно предыдущего net_worth
      - Комиссия: 0.1% от суммы сделки (слippage)
      - Нормировка цены и SMA20 по скользящему окну 252 дня
      - RSI14 нормируется в [0,1]
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100_000):
        super(StockEnvImproved, self).__init__()

        # --- 1) Вычисляем техиндикаторы и локальный минимакс (Rolling 252) ---
        df = df.copy().reset_index(drop=False)
        df["SMA20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["RollingMin"] = df["Close"].rolling(window=252, min_periods=1).min()
        df["RollingMax"] = df["Close"].rolling(window=252, min_periods=1).max()
        df = df.dropna(subset=["SMA20", "RSI14", "RollingMin", "RollingMax"]).reset_index(drop=True)

        # 2) Сохраняем индекс как DatetimeIndex (хотя он более не нужен для расчётов внутри среды)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        # 3) Массивы цен и индикаторов
        self.prices   = df["Close"].values
        self.sma20    = df["SMA20"].values
        self.rsi14    = df["RSI14"].values
        self.roll_min = df["RollingMin"].values
        self.roll_max = df["RollingMax"].values

        self.n_steps = len(self.prices)
        self.current_step = 0

        # 4) Action & Observation space
        # Действий 5: Buy20%, Buy50%, Hold, Sell20%, Sell50%
        self.action_space = spaces.Discrete(5)
        # State = [price_norm, sma_norm, rsi_norm, balance_norm, shares_norm]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # 5) Торговые параметры
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.prev_worth = initial_balance

        self.commission = 0.001  # 0.1%
        np.random.seed(0)

    def _next_observation(self) -> np.ndarray:
        price = self.prices[self.current_step]
        rmin = self.roll_min[self.current_step]
        rmax = self.roll_max[self.current_step]
        # Нормировка цены по скользящему минимуму/максимуму (252 дня)
        price_norm = (price - rmin) / (rmax - rmin + 1e-8)

        # Нормируем SMA20 по тому же скользящему окну
        sma_val = self.sma20[self.current_step]
        sma_norm = (sma_val - rmin) / (rmax - rmin + 1e-8)

        # RSI /100
        rsi_norm = self.rsi14[self.current_step] / 100.0

        balance_norm = self.balance / self.initial_balance

        # Максимальное число акций при вложении 50% баланса
        max_shares = (self.initial_balance * 0.5) / max(price, 1e-8)
        shares_norm = self.shares_held / max_shares if max_shares > 0 else 0.0

        return np.array([price_norm, sma_norm, rsi_norm, balance_norm, shares_norm], dtype=np.float32)

    def step(self, action: int):
        done = False
        price = self.prices[self.current_step]
        prev_worth = self.net_worth

        # --- Покупка 20% от баланса ---
        if action == 0:
            invest_amount = self.balance * 0.20
            # effective price с комиссией
            eff_price = price * (1 + self.commission)
            qty = int(invest_amount / eff_price)
            cost = qty * eff_price
            if qty > 0 and self.balance >= cost:
                self.balance -= cost
                self.shares_held += qty

        # --- Покупка 50% от баланса ---
        elif action == 1:
            invest_amount = self.balance * 0.50
            eff_price = price * (1 + self.commission)
            qty = int(invest_amount / eff_price)
            cost = qty * eff_price
            if qty > 0 and self.balance >= cost:
                self.balance -= cost
                self.shares_held += qty

        # --- Держать (Hold) ---
        elif action == 2:
            pass

        # --- Продажа 20% от позиции ---
        elif action == 3 and self.shares_held > 0:
            qty = max(1, int(0.20 * self.shares_held))
            eff_price = price * (1 - self.commission)
            proceeds = qty * eff_price
            self.shares_held -= qty
            self.balance += proceeds

        # --- Продажа 50% от позиции ---
        elif action == 4 and self.shares_held > 0:
            qty = max(1, int(0.50 * self.shares_held))
            eff_price = price * (1 - self.commission)
            proceeds = qty * eff_price
            self.shares_held -= qty
            self.balance += proceeds

        # Переход к следующему шагу
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        new_price = self.prices[self.current_step]
        self.net_worth = self.balance + self.shares_held * new_price

        # Процентное изменение net_worth относительно prev_worth
        reward = (self.net_worth - prev_worth) / (prev_worth + 1e-8)

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation()

    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, "
            f"Balance: {self.balance:.2f}, "
            f"Shares: {self.shares_held}, "
            f"Net worth: {self.net_worth:.2f}"
        )
