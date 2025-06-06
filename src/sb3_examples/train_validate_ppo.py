import os
import pandas as pd
import numpy as np
import torch
import random

from stable_baselines3 import PPO
from src.stock_env_improved import StockEnvImproved

# Фиксируем сиды для воспроизводимости
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def make_env_splits(csv_path: str):
    """
    Читает CSV (2010–2024), пропуская первые три служебные строки,
    задаёт имена колонок и разбивает на train/val/test по датам.
    Возвращает кортеж (train_df, val_df, test_df).
    """
    full = pd.read_csv(
        csv_path,
        skiprows=3,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"],
        index_col="Date",
        infer_datetime_format=True
    )
    train_df = full.loc["2010-01-01":"2018-12-31"].reset_index(drop=False)
    val_df   = full.loc["2019-01-01":"2020-12-31"].reset_index(drop=False)
    test_df  = full.loc["2021-01-01":"2024-12-31"].reset_index(drop=False)
    return train_df, val_df, test_df

def evaluate_on_env(env, model):
    """
    Прогоняет один полный эпизод и возвращает список net_worth во все шаги.
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
    # 1) Пути
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(BASE_DIR, "data", "AAPL_2010_2024.csv")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 2) Split data
    train_df, val_df, test_df = make_env_splits(csv_path)

    # 3) Создаём среды с улучшениями
    train_env = StockEnvImproved(df=train_df, initial_balance=100_000)
    val_env   = StockEnvImproved(df=val_df,   initial_balance=100_000)
    test_env  = StockEnvImproved(df=test_df,  initial_balance=100_000)

    # 4) Гиперпараметры PPO
    ppo_kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": os.path.join(BASE_DIR, "logs", "ppo_improved")
    }

    # 5) Обучаем PPO
    print(">>> Начинаем обучение Improved PPO (Train) …")
    model = PPO("MlpPolicy", train_env, **ppo_kwargs)
    model.learn(total_timesteps=200_000)
    model_path = os.path.join(results_dir, "ppo_improved.zip")
    model.save(model_path)
    print(">>> Модель сохранена в", model_path)

    # 6) Оценка на Validation
    print(">>> Оцениваем на Validation (2019–2020) …")
    equity_val = evaluate_on_env(val_env, model)
    final_val = equity_val[-1]
    roi_val = (final_val - 100_000) / 100_000
    daily_returns_val = np.diff(equity_val) / equity_val[:-1]
    sharpe_val = 0.0
    if np.std(daily_returns_val) != 0:
        sharpe_val = (np.mean(daily_returns_val) / np.std(daily_returns_val)) * np.sqrt(252)
    peak = equity_val[0]
    max_dd_val = 0.0
    for x in equity_val:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_dd_val:
            max_dd_val = dd

    print(f"Validation ROI:   {roi_val*100:.2f}%")
    print(f"Validation Sharpe: {sharpe_val:.2f}")
    print(f"Validation MaxDD:  {max_dd_val*100:.2f}%")
    print(f"Final net worth (Validation): {final_val:.2f}")

    # Сохраняем equity-кривую validation в CSV
    eq_val_df = pd.DataFrame({
        "step": np.arange(len(equity_val)),
        "net_worth": equity_val
    })
    eq_val_df.to_csv(os.path.join(results_dir, "equity_val_ppo.csv"), index=False)
    print(">>> Equity-кривая Validation сохранена в results/equity_val_ppo.csv")

    # 7) Оценка на Test
    print(">>> Оцениваем на Test (2021–2024) …")
    equity_test = evaluate_on_env(test_env, model)
    final_test = equity_test[-1]
    roi_test = (final_test - 100_000) / 100_000
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

    print(f"Test ROI:   {roi_test*100:.2f}%")
    print(f"Test Sharpe: {sharpe_test:.2f}")
    print(f"Test MaxDD:  {max_dd_test*100:.2f}%")
    print(f"Final net worth (Test): {final_test:.2f}")

    eq_test_df = pd.DataFrame({
        "step": np.arange(len(equity_test)),
        "net_worth": equity_test
    })
    eq_test_df.to_csv(os.path.join(results_dir, "equity_test_ppo.csv"), index=False)
    print(">>> Equity-кривая Test сохранена в results/equity_test_ppo.csv")
