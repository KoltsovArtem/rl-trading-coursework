import os
import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta  # библиотека для технических индикаторов
import torch, random

from stable_baselines3 import A2C

# -----------------------------
# 1) Фиксируем сиды для воспроизводимости
# -----------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class StockEnv(gym.Env):
    """
    Улучшенная версия окружения для торговли одной акцией (AAPL).
    - Набор действий: Discrete(3) = {0=Buy10%, 1=Hold, 2=Sell10%}
    - Состояние (обсервация): [price_norm, sma20_norm, rsi14_norm, balance_norm, shares_norm]
    - Reward: шаговая прибыль (net_worth_now − net_worth_prev) с учётом комиссии
    - Ограничение: максимум 10% баланса за одну покупку
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, csv_path, initial_balance: float = 100_000):
        super(StockEnv, self).__init__()

        # 1) Читаем CSV, конвертируем Close, высчитываем SMA20 и RSI14
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["SMA20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
        df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df = df.dropna(subset=["Close", "SMA20", "RSI14"]).reset_index(drop=True)

        # Сохраняем массивы цен и индикаторов
        self.prices = df["Close"].values
        self.sma20   = df["SMA20"].values
        self.rsi14   = df["RSI14"].values

        self.n_steps = len(self.prices)
        self.current_step = 0

        # 2) Действия: 0=Buy10%, 1=Hold, 2=Sell10%
        self.action_space = spaces.Discrete(3)

        # 3) Обсервации: пять чисел в [0,1]
        #    [price_norm, sma20_norm, rsi14_norm, balance_norm, shares_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # 4) Параметры торговли
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.prev_net_worth = initial_balance

        # 5) Комиссия и ограничитель позиции
        self.commission = 0.001               # 0.1% от объёма сделки
        self.max_investment_fraction = 0.10   # не более 10% баланса в акциях за раз

        # Фиксируем seed для action/observation
        self.action_space.seed(0)
        self.observation_space.seed(0)

    def _next_observation(self):
        """
        Формируем вектор признаков [price_norm, sma_norm, rsi_norm, balance_norm, shares_norm]
        Каждую величину нормируем к [0,1] по глобальному минимуму/максимуму.
        """
        price = self.prices[self.current_step]
        price_norm = (price - self.prices.min()) / (self.prices.max() - self.prices.min())

        sma_val = self.sma20[self.current_step]
        sma_norm = (sma_val - self.sma20.min()) / (self.sma20.max() - self.sma20.min())

        rsi_val = self.rsi14[self.current_step]
        rsi_norm = rsi_val / 100.0  # RSI в [0..100]

        balance_norm = self.balance / self.initial_balance

        # Вычисляем максимальное число акций, которое можно держать (10% от initial_balance)
        max_possible_shares = ((self.initial_balance * self.max_investment_fraction) /
                               max(price, 1e-8))
        if max_possible_shares > 0:
            shares_norm = self.shares_held / max_possible_shares
        else:
            shares_norm = 0.0

        obs = np.array([price_norm, sma_norm, rsi_norm, balance_norm, shares_norm], dtype=np.float32)
        return obs

    def step(self, action: int):
        """
        Выполняем одно действие:
        - 0 = Buy 10% от баланса (округляем вниз до целого числа акций)
        - 1 = Hold (ничего не делаем)
        - 2 = Sell 10% от позиции (округляем вниз, минимум 1 акция)
        Делаем комиссию, считаем шаговый reward = net_worth_now - prev_net_worth.
        """
        done = False
        price = self.prices[self.current_step]

        # Предыдущее net_worth, чтобы выдать шаговый reward
        prev_net = self.prev_net_worth

        # Комиссия за одну акцию
        commission_amt = price * self.commission

        # Если выбран Sell
        if action == 2 and self.shares_held > 0:
            # Продаём 10% от имеющихся акций (минимум 1 акция)
            sell_amount = max(1, int(0.10 * self.shares_held))
            proceeds = sell_amount * price - sell_amount * commission_amt
            self.shares_held -= sell_amount
            self.balance += proceeds

        # Если выбран Buy
        elif action == 0:
            # Сколько денег можно потратить: 10% от текущего баланса
            max_spend = self.balance * self.max_investment_fraction
            # Сколько акций можно купить (с учётом комиссии)
            num_to_buy = int(max_spend / (price * (1 + self.commission)))
            if num_to_buy > 0:
                cost = num_to_buy * price + num_to_buy * commission_amt
                if self.balance >= cost:
                    self.shares_held += num_to_buy
                    self.balance -= cost
        # action == 1 (Hold) → ничего не делаем

        # Переходим к следующему шагу
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        # Считаем новое net_worth после изменения цены
        new_price = self.prices[self.current_step]
        self.net_worth = self.balance + self.shares_held * new_price

        # Шаговый reward
        reward = (self.net_worth - prev_net_worth) / max(prev_net_worth, 1e-8)
        self.prev_net_worth = self.net_worth

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        """
        Сбрасываем состояние среды в начало:
        - balance = initial_balance
        - shares_held = 0
        - текущая нога (step) = 0
        - prev_net_worth = initial_balance
        """
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation()

    def render(self, mode='human'):
        print(
            f"Step: {self.current_step}, "
            f"Balance: {self.balance:.2f}, "
            f"Shares: {self.shares_held}, "
            f"Net worth: {self.net_worth:.2f}"
        )


if __name__ == "__main__":
    # -----------------------------
    # 2) Основной блок обучения
    # -----------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    aapl_csv = os.path.join(BASE_DIR, "data", "AAPL_2010_2024.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 2.1) Инициализируем окружение
    env = StockEnv(aapl_csv)

    # 2.2) Создаём модель A2C с policy Mlp (полносвязная сеть)
    model = A2C("MlpPolicy", env, verbose=1)

    # 2.3) Обучаем. Можно менять total_timesteps (например, 100_000 или 200_000)
    print(">>> Start training A2C (100k timesteps)")
    model.learn(total_timesteps=300_000)
    print(">>> Training done, saving model")
    model.save(os.path.join(results_dir, "a2c_aapl"))

    # 2.4) Оценка на одном полном эпизоде (Test → прогоним тот же датасет)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    print("Final net worth (A2C-improved):", env.net_worth)
