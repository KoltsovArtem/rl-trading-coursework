import os
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import A2C

class StockEnv(gym.Env):
    """
    Простейшее окружение: state = [normalized_price, cash, shares],
    actions = Discrete(3): Buy / Hold / Sell.
    Reward = изменение капитала.
    Для курсовой этого достаточно, можно сделать более сложное завтра.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, csv_path, initial_balance=1_000_00):
        super(StockEnv, self).__init__()
        # 1) Читаем CSV
        self.df = pd.read_csv(csv_path, parse_dates=True, index_col=0)

        # 2) Принудительно конвертируем столбец 'Close' в числа и убираем NaN
        self.df['Close'] = pd.to_numeric(self.df['Close'], errors='coerce')
        self.df = self.df.dropna(subset=['Close'])

        # 3) Извлекаем ценовой массив уже в чистом виде
        self.prices = self.df['Close'].values
        self.n_steps = len(self.prices)
        self.current_step = 0

        # Набор действий: 0=Buy, 1=Hold, 2=Sell
        self.action_space = spaces.Discrete(3)
        # Наблюдение: [цена (нормированная), cash (нормированная), shares (нормированная)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance

    def _next_observation(self):
        price = self.prices[self.current_step]
        # Нормируем цену в [0,1] по максимуму и минимуму за всё окно
        price_norm = (price - self.prices.min()) / (self.prices.max() - self.prices.min())
        balance_norm = self.balance / self.initial_balance
        shares_norm = self.shares_held / (self.initial_balance / price)
        return np.array([price_norm, balance_norm, shares_norm], dtype=np.float32)

    def step(self, action):
        done = False
        price = self.prices[self.current_step]

        # Выполнение действия
        if action == 0:  # Buy
            # Покупаем одну акцию (или максимально возможное количество)
            if self.balance >= price:
                self.shares_held += 1
                self.balance -= price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += price

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        new_price = self.prices[self.current_step]
        self.net_worth = self.balance + self.shares_held * new_price
        reward = self.net_worth - self.initial_balance  # изменение капитала от начального

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation()

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares: {self.shares_held}, Net worth: {self.net_worth}")

if __name__ == "__main__":
    # Пути
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    aapl_csv = os.path.join(BASE_DIR, "data", "AAPL_2010_2024.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Инициализируем окружение
    env = StockEnv(aapl_csv)
    model = A2C("MlpPolicy", env, verbose=1)
    # Обучаем (скорость обучения ~10^5 шагов достаточно для курсовой)
    model.learn(total_timesteps=100_000)
    model.save(os.path.join(results_dir, "a2c_aapl"))

    # Для оценки можно пройти один эпизод и вывести итоговый net_worth
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
    print("Final net worth (A2C):", env.net_worth)
