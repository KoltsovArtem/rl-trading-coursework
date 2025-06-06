import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta

class StockEnvDDPG(gym.Env):
    """
    Улучшенная среда для DDPG:
      - State: [price_norm, sma20_norm, rsi14_norm, balance_norm, shares_norm]
      - Action: Box([-1.0], [1.0]) — непрерывное число:
           action_raw <  0 → Sell долю |action_raw| от текущей позиции
           action_raw == 0 → Hold (ничего)
           action_raw >  0 → Buy долю action_raw от свободного баланса
      - Reward: процентное изменение net_worth относительно предыдущего шага
      - Комиссия: 0.1% от суммы сделки
      - Нормировка цены и SMA20 по Rolling-Min/Max за последние 252 дня
      - Техиндикаторы: SMA20, RSI14
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100_000):
        super(StockEnvDDPG, self).__init__()

        # 1) Рассчитываем SMA20, RSI14 и Rolling-Min/Max (окно 252)
        df = df.copy().reset_index(drop=False)
        df["SMA20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["RollingMin"] = df["Close"].rolling(window=252, min_periods=1).min()
        df["RollingMax"] = df["Close"].rolling(window=252, min_periods=1).max()
        df = df.dropna(subset=["SMA20", "RSI14", "RollingMin", "RollingMax"]).reset_index(drop=True)

        # Сохраняем даты (теперь Date нужен только для отладки/логов, не обязательен)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        # 2) Массивы цен и индикаторов
        self.prices   = df["Close"].values
        self.sma20    = df["SMA20"].values
        self.rsi14    = df["RSI14"].values
        self.roll_min = df["RollingMin"].values
        self.roll_max = df["RollingMax"].values

        self.n_steps = len(self.prices)
        self.current_step = 0

        # 3) Action & Observation space
        # Action: одно число ∈ [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # State: [price_norm, sma_norm, rsi_norm, balance_norm, shares_norm]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # 4) Торговые параметры
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
        # Нормировка цены по скользящему минимуму/максимуму
        price_norm = (price - rmin) / (rmax - rmin + 1e-8)

        # Нормируем SMA20 по тому же окну
        sma_val = self.sma20[self.current_step]
        sma_norm = (sma_val - rmin) / (rmax - rmin + 1e-8)

        # RSI14 / 100
        rsi_norm = self.rsi14[self.current_step] / 100.0

        balance_norm = self.balance / self.initial_balance

        # Максимальное число акций при вложении 100% баланса:
        max_shares = self.initial_balance / max(price, 1e-8)
        shares_norm = self.shares_held / max_shares if max_shares > 0 else 0.0

        return np.array([price_norm, sma_norm, rsi_norm, balance_norm, shares_norm], dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        action: массив shape=(1,) с float ∈ [-1,1]
        """
        done = False
        price = self.prices[self.current_step]
        prev_worth = self.net_worth

        action_raw = float(np.clip(action, -1.0, 1.0)[0])

        # --- Buy (action_raw > 0) ---
        if action_raw > 0:
            # Инвестируем fraction = action_raw * баланс
            invest_amount = self.balance * action_raw
            eff_price = price * (1 + self.commission)
            qty = int(invest_amount / eff_price)
            cost = qty * eff_price
            if qty > 0 and self.balance >= cost:
                self.balance -= cost
                self.shares_held += qty

        # --- Sell (action_raw < 0) ---
        elif action_raw < 0 and self.shares_held > 0:
            sell_fraction = abs(action_raw)
            qty = int(self.shares_held * sell_fraction)
            qty = max(1, qty)
            eff_price = price * (1 - self.commission)
            proceeds = qty * eff_price
            self.shares_held -= qty
            self.balance += proceeds

        # action_raw == 0 → Hold

        # Перейти к следующему шагу
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        new_price = self.prices[self.current_step]
        self.net_worth = self.balance + self.shares_held * new_price

        # Reward = процентное изменение net_worth относительно prev_worth
        reward = (self.net_worth - prev_worth) / (prev_worth + 1e-8)
        self.prev_worth = self.net_worth

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
