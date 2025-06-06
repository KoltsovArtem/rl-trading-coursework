# src/stock_env_portfolio.py

import numpy as np
import pandas as pd
import gym
from gym import spaces
import ta


class PortfolioEnv(gym.Env):
    """
    Окружение (Gym.Env) для торговли портфелем из 30 акций (DJIA).
    State = [
        balance_norm,                   # 1 число: текущий наличный баланс / initial_balance
        prices_norm[0..29],             # 30 чисел: Close-цены текущего дня / max_price_overall
        holdings_norm[0..29],           # 30 чисел: текущее кол-во акций i / max_shares
        MACD_norm[0..29],               # 30 чисел: MACD для каждого тикера (нормировано)
        RSI_norm[0..29],                # 30 чисел: RSI для каждого тикера (в [0,1])
        CCI_norm[0..29],                # 30 чисел: CCI для каждого тикера (нормировано)
        ADX_norm[0..29]                 # 30 чисел: ADX для каждого тикера (нормировано)
    ]  => всего 181 число.

    Action = вектор длины 30, каждый элемент ∈ [−1, 1]:
      action[i] > 0 ⇒ BUY акции i,
      action[i] < 0 ⇒ SELL акции i,
      action[i] = 0 ⇒ HOLD.
    Reward = Δ(net_worth) (разница чистой стоимости портфеля).
    Комиссия 0.1% за сделку. Порог Turbulence = 90-й процентиль.
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000,
        max_shares: int = 500,                   # теперь 500 по умолчанию
        commission: float = 0.001,
        price_scaling: float = None,
        indicator_scaling: dict = None,
    ):
        super().__init__()

        # 1) Проверяем DataFrame
        if df is None or df.empty:
            raise ValueError("PortfolioEnv: передан пустой или None DataFrame")

        # 2) Убедимся, что индекс – последовательный RangeIndex
        if not isinstance(df.index, pd.RangeIndex) or df.index.min() != 0 or df.index.max() != len(df)-1:
            self.df = df.reset_index(drop=True).copy()
        else:
            self.df = df.copy()

        # 3) Извлекаем список тикеров по окончанию "_Close"
        self.tickers = sorted([col.replace("_Close", "") for col in self.df.columns if col.endswith("_Close")])
        if len(self.tickers) != 30:
            raise ValueError(f"Ожидалось ровно 30 тикеров, а найдено {len(self.tickers)}: {self.tickers}")

        # 4) Основные параметры:
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.shares_owned = np.zeros(30, dtype=np.int32)
        self.net_worth = float(initial_balance)
        self.prev_net_worth = float(initial_balance)

        self.current_step = 0
        self.n_steps = len(self.df)

        # 5) Параметры сделок:
        self.max_shares = int(max_shares)
        self.commission = float(commission)

        # 6) Рассчитываем max_price (для нормировки цен):
        if price_scaling is None:
            all_prices = []
            for t in self.tickers:
                col = f"{t}_Close"
                if col not in self.df.columns:
                    raise ValueError(f"В DataFrame отсутствует колонка '{col}'")
                all_prices.append(self.df[col].values)
            all_prices = np.concatenate(all_prices)
            if all_prices.size == 0:
                raise ValueError("PortfolioEnv: массив all_prices получился пустым")
            self.max_price = float(np.nanmax(all_prices))
        else:
            self.max_price = float(price_scaling)

        # 7) Порог Turbulence = 90-й процентиль массива self.df["Turbulence"]
        if "Turbulence" not in self.df.columns:
            raise ValueError("PortfolioEnv: в DataFrame нет столбца 'Turbulence'")
        self.turbulence_threshold = float(np.percentile(self.df["Turbulence"].values, 90))

        # 8) Задаём action_space и observation_space:
        #    action_space: Box([-1,…], [1,…]) (30-мерный)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(30,), dtype=np.float32)
        #    observation_space: Box на [0,1]^181
        obs_dim = 1 + 30 + 30 + 120  # 1 баланс + 30 цен + 30 холдинги + 120 индикаторов
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # 9) Если indicator_scaling не задана, вычисляем фактические границы MACD и CCI
        if indicator_scaling is None:
            # — собираем все MACD и CCI колонки, берём глобальный min и max
            macd_cols = [f"{t}_MACD" for t in self.tickers]
            cci_cols  = [f"{t}_CCI" for t in self.tickers]

            # при этом могут быть NaN, поэтому dropna
            all_macd = self.df[macd_cols].values.reshape(-1)
            all_macd = all_macd[~np.isnan(all_macd)]
            all_cci  = self.df[cci_cols].values.reshape(-1)
            all_cci  = all_cci[~np.isnan(all_cci)]

            if all_macd.size == 0 or all_cci.size == 0:
                # если вдруг нет данных (маловероятно), оставляем константы
                self.max_macd = 200.0
                self.max_cci  = 300.0
            else:
                # из фактического диапазона берём самое большое абсолютное значение
                self.max_macd = float(np.max(np.abs(all_macd)))    # чтобы симметрично вокруг 0
                self.max_cci  = float(np.max(np.abs(all_cci)))     # симметрично вокруг 0

            # RSI и ADX мы знаем, что всегда в [0..100], оставляем
            self.max_rsi  = 100.0
            self.max_adx  = 100.0
        else:
            # если заданы вручную, просто приводим к float
            self.max_macd = float(indicator_scaling.get("macd", 200.0))
            self.max_rsi  = float(indicator_scaling.get("rsi", 100.0))
            self.max_cci  = float(indicator_scaling.get("cci", 300.0))
            self.max_adx  = float(indicator_scaling.get("adx", 100.0))

        # 10) Сеяем "random seed" для numpy (если внутри есть np.random)
        np.random.seed(0)

    def _get_price_vector(self, step: int) -> np.ndarray:
        """Возвращаем вектор из 30 Close-цен для тикеров на шаге step."""
        prices = [self.df.loc[step, f"{t}_Close"] for t in self.tickers]
        return np.array(prices, dtype=np.float32)

    def _get_indicator_vectors(self, step: int) -> np.ndarray:
        """
        Возвращаем вектор из 120 индикаторов:
        [MACD(30), RSI(30), CCI(30), ADX(30)], каждый нормируем в [0,1].
        """
        macd, rsi, cci, adx = [], [], [], []

        for t in self.tickers:
            # Проверяем наличие колонок
            for suffix in ["_MACD", "_RSI", "_CCI", "_ADX"]:
                col_name = f"{t}{suffix}"
                if col_name not in self.df.columns:
                    raise ValueError(f"В DataFrame отсутствует колонка '{col_name}'")
            macd.append(self.df.loc[step, f"{t}_MACD"])
            rsi.append(self.df.loc[step, f"{t}_RSI"])
            cci.append(self.df.loc[step, f"{t}_CCI"])
            adx.append(self.df.loc[step, f"{t}_ADX"])

        # Нормируем MACD и CCI относительно вычисленных max_abs
        macd = np.array(macd, dtype=np.float32)
        if self.max_macd > 0:
            macd = (macd / self.max_macd) / 2.0 + 0.5   # (–max..+max) → (0..1)
        else:
            macd = 0.5 * np.ones_like(macd)             # если вдруг max_macd=0

        cci = np.array(cci, dtype=np.float32)
        if self.max_cci > 0:
            cci = (cci / self.max_cci) / 2.0 + 0.5      # (–max..+max) → (0..1)
        else:
            cci = 0.5 * np.ones_like(cci)

        # RSI и ADX нормируем «как раньше»
        rsi = np.array(rsi, dtype=np.float32) / self.max_rsi   # (0..100) → (0..1)
        adx = np.array(adx, dtype=np.float32) / self.max_adx   # (0..100) → (0..1)

        # Clip, чтобы гарантировать [0,1]
        macd = np.clip(macd, 0.0, 1.0)
        rsi  = np.clip(rsi,  0.0, 1.0)
        cci  = np.clip(cci,  0.0, 1.0)
        adx  = np.clip(adx,  0.0, 1.0)

        return np.concatenate([macd, rsi, cci, adx], axis=0).astype(np.float32)  # shape=(120,)

    def _get_turbulence(self, step: int) -> float:
        """Возвращаем значение Turbulence на данном шаге."""
        return float(self.df.loc[step, "Turbulence"])

    def _next_observation(self) -> np.ndarray:
        """
        Собираем state (181-мерный вектор):
        [balance_norm(1), prices_norm(30), holdings_norm(30), indicators_norm(120)].
        """
        step = self.current_step

        # 1) Нормируем баланс:
        balance_norm = np.array([self.balance / self.initial_balance], dtype=np.float32)

        # 2) Нормируем цены:
        prices = self._get_price_vector(step)
        prices_norm = (prices / (self.max_price + 1e-8)).astype(np.float32)

        # 3) Нормируем холдинги:
        holdings_norm = (self.shares_owned.astype(np.float32) / float(self.max_shares + 1e-8))
        holdings_norm = np.clip(holdings_norm, 0.0, 1.0).astype(np.float32)

        # 4) Собираем индикаторы:
        indicators_norm = self._get_indicator_vectors(step)  # shape=(120,)

        return np.concatenate([
            balance_norm,      # (1,)
            prices_norm,       # (30,)
            holdings_norm,     # (30,)
            indicators_norm    # (120,)
        ], axis=0)

    def step(self, action: np.ndarray):
        """
        Шаг перехода:
          1) Если Turbulence > threshold → force sell всех 30 позиций.
          2) Иначе: desired_shares = action * max_shares; trades = desired_shares - owned.
             Сначала SELL, затем BUY, с учётом комиссии 0.1%.
          3) current_step += 1; считаем net_worth; reward = Δ(net_worth).
        """
        done = False
        step = self.current_step

        # 1) Сохраняем текущую стоимость портфеля:
        prev_worth = float(self.balance + np.dot(self._get_price_vector(step), self.shares_owned))

        # 2) Проверяем Turbulence:
        if self._get_turbulence(step) > self.turbulence_threshold:
            # Force-sell all:
            current_prices = self._get_price_vector(step)
            proceeds = float(np.dot(current_prices, self.shares_owned))
            commission_amt = proceeds * self.commission
            self.balance += (proceeds - commission_amt)
            self.shares_owned = np.zeros_like(self.shares_owned)
        else:
            # 3) Интерпретируем action ∈ [−1,1] → desired_shares ∈ [−max_shares, +max_shares]
            desired_shares = (action * float(self.max_shares)).astype(np.int32)
            desired_shares = np.clip(desired_shares, -self.max_shares, +self.max_shares)

            # 4) trades = desired_shares − owned
            trades = desired_shares - self.shares_owned
            prices = self._get_price_vector(step)

            # a) SELL (trades < 0)
            sell_indices = np.where(trades < 0)[0]
            for i in sell_indices:
                amount_to_sell = int(-trades[i])
                if amount_to_sell <= 0:
                    continue
                if self.shares_owned[i] >= amount_to_sell:
                    sale_proceeds = float(amount_to_sell) * prices[i]
                    commission_amt = sale_proceeds * self.commission
                    self.balance += (sale_proceeds - commission_amt)
                    self.shares_owned[i] -= amount_to_sell
                else:
                    # Если меньше акций, чем хотели продать, продаём всё, что есть
                    actual = int(self.shares_owned[i])
                    if actual > 0:
                        sale_proceeds = float(actual) * prices[i]
                        commission_amt = sale_proceeds * self.commission
                        self.balance += (sale_proceeds - commission_amt)
                    self.shares_owned[i] = 0

            # b) BUY (trades > 0)
            buy_indices = np.where(trades > 0)[0]
            for i in buy_indices:
                amount_to_buy = int(trades[i])
                if amount_to_buy <= 0:
                    continue
                total_cost = float(amount_to_buy) * prices[i]
                commission_amt = total_cost * self.commission
                cost_with_comm = total_cost + commission_amt
                if self.balance >= cost_with_comm:
                    self.balance -= cost_with_comm
                    self.shares_owned[i] += amount_to_buy
                else:
                    # Недостаточно средств — пропускаем этот ордер
                    pass

        # 5) Переходим к следующему шагу
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        # 6) Считаем новую net_worth
        next_step = min(self.current_step, self.n_steps - 1)
        new_prices = self._get_price_vector(next_step)
        self.net_worth = float(self.balance + np.dot(new_prices, self.shares_owned))

        # 7) Reward = Δ(net_worth)
        reward = float(self.net_worth - prev_worth)

        # 8) Собираем новое наблюдение
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        """
        Сбрасываем среду:
          balance = initial_balance, shares_owned=0, current_step=0, net_worth=initial_balance
        """
        self.balance = float(self.initial_balance)
        self.shares_owned = np.zeros(30, dtype=np.int32)
        self.net_worth = float(self.initial_balance)
        self.prev_net_worth = float(self.initial_balance)
        self.current_step = 0
        return self._next_observation()

    def render(self, mode='human'):
        """
        Для отладки печатаем баланс, net_worth и Holdings.
        """
        prices = self._get_price_vector(self.current_step)
        print(
            f"Step: {self.current_step} | "
            f"Balance: {self.balance:.2f} | "
            f"Net worth: {self.net_worth:.2f} | "
            f"Holdings: {self.shares_owned}"
        )
