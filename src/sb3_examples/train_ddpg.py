import os
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, csv_path, initial_balance=1_000_00):
        super(StockEnv, self).__init__()
        self.df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        self.prices = self.df['Close'].values
        self.n_steps = len(self.prices)
        self.current_step = 0

        # У DDPG action_space должен быть Box (континуальный), но мы оставим Discrete для упрощения:
        # Для настоящего DDPG нужно изменить логику: например, action ∈ [−1,1] для пропорции покупки/продажи.
        # Здесь оставим дискретный каркас, и SB3 сам приведет его к Box.
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance

    def _next_observation(self):
        price = self.prices[self.current_step]
        price_norm = (price - self.prices.min()) / (self.prices.max() - self.prices.min())
        balance_norm = self.balance / self.initial_balance
        shares_norm = self.shares_held / (self.initial_balance / price)
        return np.array([price_norm, balance_norm, shares_norm], dtype=np.float32)

    def step(self, action):
        done = False
        price = self.prices[self.current_step]
        # Для DDPG будем считать действие как float, но здесь просто приведём к int
        action = int(action)
        if action == 0 and self.balance >= price:
            self.shares_held += 1
            self.balance -= price
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += price

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        new_price = self.prices[self.current_step]
        self.net_worth = self.balance + self.shares_held * new_price
        reward = self.net_worth - self.initial_balance

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
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    aapl_csv = os.path.join(BASE_DIR, "data", "AAPL_2010_2024.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    env = StockEnv(aapl_csv)
    # Добавляем гауссов шум для DDPG
    n_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save(os.path.join(results_dir, "ddpg_aapl"))

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    print("Final net worth (DDPG):", env.net_worth)
