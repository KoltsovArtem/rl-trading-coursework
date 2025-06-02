import pandas as pd
import numpy as np
import os

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td = reward + self.gamma * best_next - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td

def discretize_price(price, price_min, price_max, n_bins):
    """
    Простая дискретизация цены в n_bins.
    Возвращает индекс bin-а для данного price.
    """
    bins = np.linspace(price_min, price_max, n_bins + 1)
    return np.digitize(price, bins) - 1  # индексы от 0 до n_bins-1

def run_q_learning(csv_path, n_bins=10, n_epochs=5):
    """
    Загружает исторические данные, дискретизирует цену в n_bins,
    запускает несколько эпох Q-learning и возвращает итоговую Q-таблицу.
    """
    # 1) Чтение данных
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    # Берём только колонку 'Close'
    prices = df['Close'].values
    price_min, price_max = prices.min(), prices.max()
    n_states = n_bins
    n_actions = 3  # 0=Buy, 1=Hold, 2=Sell

    agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)

    for epoch in range(n_epochs):
        state = discretize_price(prices[0], price_min, price_max, n_bins)
        for t in range(0, len(prices)-1):
            action = agent.choose_action(state)
            next_price = prices[t+1]
            next_state = discretize_price(next_price, price_min, price_max, n_bins)

            # Простейшая функция награды: если продал при росте — +1, если купил при падении — -1, иначе 0
            reward = 0
            if action == 0 and next_price > prices[t]:   # Buy and price went up
                reward = 1
            elif action == 2 and next_price < prices[t]: # Sell and price went down
                reward = 1
            elif action == 0 and next_price < prices[t]: # Buy and price went down
                reward = -1
            elif action == 2 and next_price > prices[t]: # Sell and price went up
                reward = -1

            agent.update(state, action, reward, next_state)
            state = next_state

        print(f"Epoch {epoch+1}/{n_epochs} completed.")

    return agent

if __name__ == "__main__":
    # Предполагается, что файл AAPL_2010_2024.csv лежит в папке data/
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    aapl_csv = os.path.join(BASE_DIR, "data", "AAPL_2010_2024.csv")

    if not os.path.exists(aapl_csv):
        raise FileNotFoundError(f"Не нашёл файл {aapl_csv}")

    agent = run_q_learning(aapl_csv, n_bins=10, n_epochs=5)
    # Здесь можно добавить сохранение agent.q_table в CSV или визуализацию
    np.savetxt(os.path.join(BASE_DIR, "results", "q_table_aapl.csv"), agent.q_table, delimiter=",")
    print("Q-таблица сохранена в results/q_table_aapl.csv")
