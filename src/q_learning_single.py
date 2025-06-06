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
        # ε-жадная стратегия
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td = reward + self.gamma * best_next - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td


def discretize_features(features, bins_per_feature):
    """
    Дискретизация векторов features (list из F признаков) в единственный индекс:
      - Для каждого признака i мы заранее знаем его min и max (см. ниже в run_q_learning)
      - Затем делаем np.digitize, получая целое bin_i ∈ [0, bins_per_feature-1].
      - Кодируем их в одно число: idx = bin_0 + bin_1 * n + bin_2 * n^2 + bin_3 * n^3, ...
    """
    idx = 0
    for i, b in enumerate(features):
        idx += b * (bins_per_feature ** i)
    return int(idx)


def compute_rolling_indicators(prices, window=20, bollinger_k=2):
    """
    На входе:
      prices — numpy-массив цен (Close).
    Возвращаем кортеж из четырёх numpy-массивов той же длины:
      price_arr, sma_arr, boll_upper_arr, boll_lower_arr
    где:
      sma_arr[i] = среднее за window последних дней, если i>=window-1, иначе NaN
      std_arr аналогично (для Bollinger)
      boll_upper = sma + k·std; boll_lower = sma − k·std
      price_arr просто копирует prices
    """
    price_arr = prices.copy()
    sma = pd.Series(prices).rolling(window=window, min_periods=1).mean().to_numpy()
    std = pd.Series(prices).rolling(window=window, min_periods=1).std(ddof=0).fillna(0).to_numpy()
    boll_upper = sma + bollinger_k * std
    boll_lower = sma - bollinger_k * std
    return price_arr, sma, boll_upper, boll_lower


def run_q_learning(csv_path, n_bins=10, n_epochs=5, window=20, boll_k=2):
    """
    Основная функция:
      1) Считывает данные AAPL.
      2) Вычисляет SMA20 и линии Боллинджера (k=2) на основе window=20.
      3) Готовит параметры дискретизации (min/max по каждому из четырёх признаков).
      4) Создаёт агента: n_states = n_bins^4, n_actions = 3.
      5) Тренирует Q на n_epochs, каждый раз делая 4 прохода по серии (четыре inner-итерации).
         При этом поддерживается баланс и количество акций: можно держать не более 1 акции.
         Награда:
           - Если action == SELL и shares_held == 0 → reward = −100 (штраф).
           - Если action == BUY и shares_held == 0 → покупаем 1 акцию по цене current_price, reward = 0.
           - Если action == BUY и shares_held == 1 → просто reward = 0 (не покупаем вторую).
           - Если action == SELL и shares_held == 1 → продаём, reward = price_now − buy_price.
           - При action == HOLD:
               • Если shares_held == 1 → reward = price_now − price_prev (динамика цены).
               • Иначе reward = 0.
      6) После тренировки сохраняет Q-таблицу в CSV.
      7) Запускает Random Walk тест (1000 симуляций):
         • Для каждой симуляции формируется случайная ценовая серия длины T такой же, как original.
         • Агент обучается на ней точно так же (n_epochs итераций), после чего симулируем
           финальный баланс и сохраняем прибыль. В конце выводим среднюю прибыль по 1000.
    """
    # 1) Считываем CSV
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    prices_full = df['Close'].values
    T = len(prices_full)

    # 2) Вычисляем rolling-индикаторы (SMA20 + Bollinger)
    price_arr, sma_arr, boll_up_arr, boll_low_arr = compute_rolling_indicators(prices_full, window=window,
                                                                               bollinger_k=boll_k)

    # 3) Определяем min/max для каждой из 4 признаков (для binning)
    #    Берём весь отрезок, даже первые window дней, т.к. rolling с min_periods=1 даёт значения
    p_min, p_max = np.nanmin(price_arr), np.nanmax(price_arr)
    sma_min, sma_max = np.nanmin(sma_arr), np.nanmax(sma_arr)
    bu_min, bu_max = np.nanmin(boll_up_arr), np.nanmax(boll_up_arr)
    bl_min, bl_max = np.nanmin(boll_low_arr), np.nanmax(boll_low_arr)

    # 4-мерный state: (price, sma, boll_up, boll_low), каждый дискретизируем отдельно
    #    Для binning используем np.linspace(min, max, n_bins+1) → np.digitize → [0..n_bins-1]

    def discretize_quad(i):
        """
        Вернёт одно число состояния на основе 4 признаков в момент i.
        """
        # 4 признака
        f0 = price_arr[i]
        f1 = sma_arr[i]
        f2 = boll_up_arr[i]
        f3 = boll_low_arr[i]

        # Для каждого признака находим bin_i ∈ [0..n_bins-1]
        bin0 = np.digitize(f0, np.linspace(p_min, p_max, n_bins + 1)) - 1
        bin1 = np.digitize(f1, np.linspace(sma_min, sma_max, n_bins + 1)) - 1
        bin2 = np.digitize(f2, np.linspace(bu_min, bu_max, n_bins + 1)) - 1
        bin3 = np.digitize(f3, np.linspace(bl_min, bl_max, n_bins + 1)) - 1

        # clip each bin, чтобы не выйти за границы
        bin0 = int(np.clip(bin0, 0, n_bins - 1))
        bin1 = int(np.clip(bin1, 0, n_bins - 1))
        bin2 = int(np.clip(bin2, 0, n_bins - 1))
        bin3 = int(np.clip(bin3, 0, n_bins - 1))

        # Кодируем 4 бин-индекса в одно число: bin0 + bin1*n_bins + bin2*(n_bins^2) + bin3*(n_bins^3)
        return discretize_features([bin0, bin1, bin2, bin3], n_bins)

    # Всего состояний:
    n_states = n_bins ** 4
    n_actions = 3  # {0=Buy,1=Hold,2=Sell}

    # 5) Инициализируем агента
    agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.3)

    # Здесь храним баланс и владение одной акцией (максимум 1), для корректного reward
    initial_balance = 100_000.0
    # Но это нужно только для тренировки на реальной серии, для random walk теста заведём свои переменные

    for epoch in range(n_epochs):
        # Повторяем 4 раза обновление Q на одном одном train-эпизоде
        for inner_iter in range(4):
            balance = initial_balance
            shares_held = 0  # 0 или 1
            buy_price = 0.0  # цена покупки для расчёта выгоды при продаже
            prev_price = price_arr[0]  # цена предыдущего шага (для reward при Hold)

            # Начальное состояние
            state = discretize_quad(0)

            for t in range(0, T - 1):
                action = agent.choose_action(state)
                price_now = price_arr[t]
                price_next = price_arr[t + 1]
                next_state = discretize_quad(t + 1)

                reward = 0.0

                # --- SELL ---
                if action == 2:
                    if shares_held == 0:
                        # штраф −100 за попытку продать без акций
                        reward = -100.0
                    else:
                        # продаём 1 акцию
                        reward = price_now - buy_price
                        balance += price_now
                        shares_held = 0

                # --- BUY ---
                elif action == 0:
                    if shares_held == 0 and balance >= price_now:
                        shares_held = 1
                        buy_price = price_now
                        balance -= price_now
                        reward = 0.0  # как в ВКР: при покупке reward = 0
                    else:
                        # либо уже есть акция, либо нет денег → считаем как Hold (reward ниже)
                        if shares_held == 1:
                            reward = price_now - prev_price
                        else:
                            reward = 0.0

                # --- HOLD ---
                else:  # action == 1
                    if shares_held == 1:
                        reward = price_now - prev_price
                    else:
                        reward = 0.0

                # Обновляем Q-таблицу
                agent.update(state, action, reward, next_state)

                # Перейдём дальше
                state = next_state
                prev_price = price_now

        print(f"Epoch {epoch + 1}/{n_epochs} completed.")

    # После тренировки на реальной серии сохраняем Q-таблицу
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    np.savetxt(os.path.join(results_dir, "q_table_aapl.csv"), agent.q_table, delimiter=",")
    print("Q-таблица сохранена в results/q_table_aapl.csv")

    # 6) Random Walk-тест
    #    Генерируем 1000 серий «цена[t+1] = цена[t] * (1 + δ)», δ ∈ Uniform(-0.01, +0.01)
    #    На каждой серии обучаем нового агента (тот же n_states, n_actions), n_epochs эпох, 4 inner-итерации,
    #    а затем симулируем финальный баланс (не обновляя Q, просто следуем policy).
    print("\n=== Random Walk Test (1000 sims) ===")
    profits = []
    length = T
    for sim in range(1000):
        print("Sim ", sim + 1)
        # Генерируем случайную цену
        rw_prices = np.zeros(length)
        rw_prices[0] = prices_full[0]  # начнём с реальной первой цены, но можно взять 100
        for t in range(1, length):
            delta = np.random.uniform(-0.01, 0.01)
            rw_prices[t] = rw_prices[t - 1] * (1 + delta)

        # Вычисляем индикаторы для этой случайной серии
        rw_price_arr, rw_sma, rw_bu, rw_bl = compute_rolling_indicators(rw_prices, window=window, bollinger_k=boll_k)

        # Берём min/max для биннинга из той же логики:
        rp_min, rp_max = np.nanmin(rw_price_arr), np.nanmax(rw_price_arr)
        rs_min, rs_max = np.nanmin(rw_sma), np.nanmax(rw_sma)
        rbu_min, rbu_max = np.nanmin(rw_bu), np.nanmax(rw_bu)
        rbl_min, rbl_max = np.nanmin(rw_bl), np.nanmax(rw_bl)

        def discretize_quad_rw(i):
            f0 = rw_price_arr[i]
            f1 = rw_sma[i]
            f2 = rw_bu[i]
            f3 = rw_bl[i]
            b0 = np.digitize(f0, np.linspace(rp_min, rp_max, n_bins + 1)) - 1
            b1 = np.digitize(f1, np.linspace(rs_min, rs_max, n_bins + 1)) - 1
            b2 = np.digitize(f2, np.linspace(rbu_min, rbu_max, n_bins + 1)) - 1
            b3 = np.digitize(f3, np.linspace(rbl_min, rbl_max, n_bins + 1)) - 1
            b0 = int(np.clip(b0, 0, n_bins - 1))
            b1 = int(np.clip(b1, 0, n_bins - 1))
            b2 = int(np.clip(b2, 0, n_bins - 1))
            b3 = int(np.clip(b3, 0, n_bins - 1))
            return discretize_features([b0, b1, b2, b3], n_bins)

        # 6.1) Обучаем нового агента на random walk
        agent_rw = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.3)
        for epoch in range(n_epochs):
            print("Epoch ", epoch + 1)
            for inner_iter in range(4):
                shares_held = 0
                buy_price = 0.0
                prev_price = rw_price_arr[0]
                state_rw = discretize_quad_rw(0)
                for t in range(0, length - 1):
                    action = agent_rw.choose_action(state_rw)
                    price_now = rw_price_arr[t]
                    price_next = rw_price_arr[t + 1]
                    next_state_rw = discretize_quad_rw(t + 1)

                    reward = 0.0
                    # SELL
                    if action == 2:
                        if shares_held == 0:
                            reward = -100.0
                        else:
                            reward = price_now - buy_price
                            shares_held = 0
                    # BUY
                    elif action == 0:
                        if shares_held == 0:
                            shares_held = 1
                            buy_price = price_now
                            reward = 0.0
                        else:
                            reward = (price_now - prev_price) if shares_held == 1 else 0.0
                    # HOLD
                    else:
                        reward = (price_now - prev_price) if shares_held == 1 else 0.0

                    agent_rw.update(state_rw, action, reward, next_state_rw)
                    state_rw = next_state_rw
                    prev_price = price_now

        # 6.2) После тренировки симулируем один проход «deterministic policy» (без ε-исследования),
        #      чтобы подсчитать финальную прибыль (снова track balance/shares):
        balance = initial_balance
        shares_held = 0
        buy_price = 0.0
        prev_price = rw_price_arr[0]
        state_rw = discretize_quad_rw(0)
        for t in range(0, length - 1):
            # deterministic = True → always берем argmax без шума
            action = np.argmax(agent_rw.q_table[state_rw])
            price_now = rw_price_arr[t]
            # Применяем ту же логику reward-теста, но здесь нам важен только итоговый баланс
            if action == 2:
                if shares_held == 1:
                    balance += price_now
                    shares_held = 0
            elif action == 0:
                if shares_held == 0 and balance >= price_now:
                    shares_held = 1
                    buy_price = price_now
                    balance -= price_now
            # HOLD — ничего не делаем
            # Переход к следующему состоянию
            state_rw = discretize_quad_rw(t + 1)

        # В конце, если последний шаг остался с акцией, продадим её по последней цене
        if shares_held == 1:
            balance += rw_price_arr[-1]
            shares_held = 0

        profit = balance - initial_balance
        profits.append(profit)

    avg_profit = np.mean(profits)
    print(f"Random Walk: average profit over 1000 sims = {avg_profit:.2f}")


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    aapl_csv = os.path.join(BASE_DIR, "data", "AAPL_2010_2024.csv")

    if not os.path.exists(aapl_csv):
        raise FileNotFoundError(f"Не нашел файл {aapl_csv}")

    run_q_learning(aapl_csv, n_bins=10, n_epochs=5)
