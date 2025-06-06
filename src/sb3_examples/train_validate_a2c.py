import os
import pandas as pd
import numpy as np
import torch
import random

# Импорт вашего класса StockEnv из файла stock_env.py
from src.stock_env import StockEnv
from stable_baselines3 import A2C

# Фиксируем сиды для воспроизводимости
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def make_env_splits(csv_path: str):
    """
    Читает полный CSV и разбивает на три DataFrame: Train, Val, Test.
    Возвращает кортеж (train_df, val_df, test_df).
    """
    full = pd.read_csv(csv_path, skiprows=3,
                       names=["Date", "Close", "High", "Low", "Open", "Volume"],
                       parse_dates=["Date"], index_col="Date",
                       infer_datetime_format=True)
    # full.index — это DatetimeIndex
    train_df = full.loc["2010-01-01":"2018-12-31"].copy().reset_index(drop=True)
    val_df   = full.loc["2019-01-01":"2020-12-31"].copy().reset_index(drop=True)
    test_df  = full.loc["2021-01-01":"2024-12-31"].copy().reset_index(drop=True)
    return train_df, val_df, test_df

def evaluate_on_env(env, model):
    """
    Прогоняет один полный эпизод: возвращает список equity (нетто-стоимости) по всем шагам.
    """
    obs = env.reset()
    done = False
    equity_curve = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        equity_curve.append(env.net_worth)
    return equity_curve

if __name__ == "__main__":
    # 1) Пути к файлам и папкам
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # BASE_DIR указывает на корень rl-trading-coursework
    csv_path = os.path.join(BASE_DIR, "data", "AAPL_2010_2024.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 2) Создание DataFrame для Train/Val/Test
    train_df, val_df, test_df = make_env_splits(csv_path)

    # 3) Инициализируем среды (передаём соответствующие DataFrame)
    train_env = StockEnv(df=train_df, initial_balance=100_000)
    val_env   = StockEnv(df=val_df,   initial_balance=100_000)
    test_env  = StockEnv(df=test_df,  initial_balance=100_000)

    # 4) Задаём гиперпараметры для A2C
    total_timesteps = 100_000
    a2c_kwargs = dict(
        learning_rate = 7e-4,
        gamma = 0.99,
        ent_coef = 0.01,
        verbose = 1
    )

    # 5) Обучаем модель A2C на тренировочных данных
    print(">>> Начинаем обучение A2C (Train) …")
    model = A2C("MlpPolicy", train_env, **a2c_kwargs)
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(results_dir, "a2c_train.zip")
    model.save(model_path)
    print(">>> Модель сохранена в", model_path)

    # 6) Оценка модели на валидации
    print(">>> Оцениваем на Validation …")
    equity_val = evaluate_on_env(val_env, model)

    # Присваиваем финальное значение net_worth
    final_net_val = equity_val[-1]

    # Считаем ROI, Sharpe, MaxDrawdown
    roi_val = (final_net_val - 100_000) / 100_000
    daily_returns = np.diff(equity_val) / equity_val[:-1]
    sharpe_val = 0.0
    if np.std(daily_returns) != 0:
        sharpe_val = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    peak = equity_val[0]
    max_drawdown = 0.0
    for x in equity_val:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    # Выводим метрики и финальный баланс
    print(f"Validation ROI:   {roi_val*100:.2f}%")
    print(f"Validation Sharpe: {sharpe_val:.2f}")
    print(f"Validation MaxDD:  {max_drawdown*100:.2f}%")
    print(f"Final net worth (Validation): {final_net_val:.2f}")

    # Сохраняем equity-кривую в CSV
    eq_df = pd.DataFrame({
        "step": np.arange(len(equity_val)),
        "net_worth": equity_val
    })
    eq_df.to_csv(os.path.join(results_dir, "equity_val_a2c.csv"), index=False)
    print(">>> Equity-кривая сохранена в results/equity_val_a2c.csv")

    # 7) Тестирование на тестовом наборе
    print(">>> Оцениваем на Test …")
    equity_test = evaluate_on_env(test_env, model)

    # Присваиваем финальное значение net_worth для теста
    final_net_test = equity_test[-1]

    roi_test = (final_net_test - 100_000) / 100_000
    daily_returns_test = np.diff(equity_test) / equity_test[:-1]
    sharpe_test = 0.0
    if np.std(daily_returns_test) != 0:
        sharpe_test = (np.mean(daily_returns_test) / np.std(daily_returns_test)) * np.sqrt(252)
    peak = equity_test[0]
    max_dd_test = 0.0
    for x in equity_test:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_dd_test:
            max_dd_test = dd

    # Выводим метрики и финальный баланс для теста
    print(f"Test ROI:   {roi_test*100:.2f}%")
    print(f"Test Sharpe: {sharpe_test:.2f}")
    print(f"Test MaxDD:  {max_dd_test*100:.2f}%")
    print(f"Final net worth (Test): {final_net_test:.2f}")

    equity_test_df = pd.DataFrame({
        "step": np.arange(len(equity_test)),
        "net_worth": equity_test
    })
    equity_test_df.to_csv(os.path.join(results_dir, "equity_test_a2c.csv"), index=False)
    print(">>> Equity-кривая для Test сохранена в results/equity_test_a2c.csv")
