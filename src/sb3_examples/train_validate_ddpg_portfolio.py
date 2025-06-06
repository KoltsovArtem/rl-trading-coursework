# src/sb3_examples/train_validate_ddpg_portfolio.py

import os
import glob
import numpy as np
import pandas as pd
import torch
import random

from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Берём то же PortfolioEnv, что и для A2C, но теперь для DDPG на 30 акциях
from src.stock_env_portfolio import PortfolioEnv

# Фиксируем сиды для воспроизводимости
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def load_and_preprocess_djia(data_folder: str):
    """
    1) Считываем все CSV-файлы из data_folder/*.csv.
    2) Пропускаем первые три строки ("Price,…", "Ticker,…", "Date,,,,,") и задаём
       имена колонок ["Date","Close","High","Low","Open","Volume"].
    3) parse_dates=["Date"], index_col="Date" — чтобы Date сразу стал DatetimeIndex.
    4) Оставляем только ['Open','High','Low','Close','Volume'], переименовывая в
       '<TICKER>_Open', '<TICKER>_Close' и т.д.
    5) Inner-join по индексу Date (оставляем только даты, где торгуются все тикеры).
    6) Считаем технические индикаторы (MACD, RSI, CCI, ADX) для каждого тикера.
    7) Считаем Turbulence (скользящее окно 60 дней) по доходностям Close.
    8) Dropna() и reset_index(), чтобы 'Date' снова стала обычным столбцом.
    Возвращаем: df_full (с колонкой 'Date') и список tickers.
    """
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    all_files.sort()

    list_df = []
    tickers = []

    for filepath in all_files:
        ticker = os.path.basename(filepath).replace(".csv", "")
        tickers.append(ticker)

        # 1) Читаем CSV, пропуская первые три «служебных» строки
        df_t = pd.read_csv(
            filepath,
            skiprows=3,
            names=["Date", "Close", "High", "Low", "Open", "Volume"],
            usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
            parse_dates=["Date"],
            index_col="Date"
        )

        # 2) Проверяем обязательные колонки
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df_t.columns:
                raise ValueError(f"В файле {filepath} отсутствует столбец '{col}' после skiprows.")

        # 3) Переименовываем
        df_t = df_t.rename(columns={
            "Open":   f"{ticker}_Open",
            "High":   f"{ticker}_High",
            "Low":    f"{ticker}_Low",
            "Close":  f"{ticker}_Close",
            "Volume": f"{ticker}_Volume"
        })

        list_df.append(df_t)

    # 4) Inner-join всех датафреймов по индексу Date
    df_all = pd.concat(list_df, axis=1, join="inner").sort_index()

    # 5) Считаем технические индикаторы для каждого тикера
    macd_list = []
    rsi_list  = []
    cci_list  = []
    adx_list  = []

    for ticker in tickers:
        close = df_all[f"{ticker}_Close"]
        high  = df_all[f"{ticker}_High"]
        low   = df_all[f"{ticker}_Low"]

        macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        macd_list.append(macd_ind.macd_diff().rename(f"{ticker}_MACD"))

        rsi_ind = RSIIndicator(close, window=14)
        rsi_list.append(rsi_ind.rsi().rename(f"{ticker}_RSI"))

        cci_ind = CCIIndicator(high, low, close, window=14)
        cci_list.append(cci_ind.cci().rename(f"{ticker}_CCI"))

        adx_ind = ADXIndicator(high, low, close, window=14)
        adx_list.append(adx_ind.adx().rename(f"{ticker}_ADX"))

    df_indicators = pd.concat([*macd_list, *rsi_list, *cci_list, *adx_list], axis=1)

    # 6) Объединяем цены и индикаторы в один DataFrame
    df_full = pd.concat([df_all, df_indicators], axis=1)

    # 7) Считаем Turbulence (скользящее окно 60 дней по доходностям)
    close_cols = [f"{ticker}_Close" for ticker in tickers]
    returns = df_full[close_cols].pct_change()

    turb_values = [0.0] * len(df_full)
    for i in range(60, len(df_full)):
        window_rets = returns.iloc[i - 60 : i]
        if window_rets.isnull().values.any():
            turb_values[i] = 0.0
            continue
        mu  = window_rets.mean(axis=0).values.reshape(-1, 1)
        cov = window_rets.cov().values
        r_t = returns.iloc[i].values.reshape(-1, 1)
        try:
            inv_cov = np.linalg.inv(cov)
            delta = r_t - mu
            turb_values[i] = float(delta.T.dot(inv_cov).dot(delta))
        except np.linalg.LinAlgError:
            turb_values[i] = 0.0

    df_full["Turbulence"] = turb_values

    # 8) Убираем NaN (индикаторы и первые 60 строк Turbulence) и сбрасываем индекс
    df_full = df_full.dropna().reset_index()  # 'Date' становится обычной колонкой

    return df_full, tickers


def make_portfolio_env_splits(data_folder: str):
    """
    1) Загружаем и предобрабатываем всё через load_and_preprocess_djia().
    2) Разбиваем на три периода:
         • train_df  (2010-01-01 — 2018-12-31)
         • val_df    (2019-01-01 — 2020-12-31)
         • test_df   (2021-01-01 — 2024-12-31)
    3) Возвращаем train_df, val_df, test_df, tickers.
    """
    df_full, tickers = load_and_preprocess_djia(data_folder)

    df_full["Date"] = pd.to_datetime(df_full["Date"])
    df_full = df_full.sort_values("Date")

    train_df = df_full.loc[
        (df_full["Date"] >= "2010-01-01") & (df_full["Date"] <= "2018-12-31")
    ].copy().reset_index(drop=True)

    val_df = df_full.loc[
        (df_full["Date"] >= "2019-01-01") & (df_full["Date"] <= "2020-12-31")
    ].copy().reset_index(drop=True)

    test_df = df_full.loc[
        (df_full["Date"] >= "2021-01-01") & (df_full["Date"] <= "2024-12-31")
    ].copy().reset_index(drop=True)

    return train_df, val_df, test_df, tickers


def evaluate_on_env(env, model):
    """
    Прогоняет один полный эпизод на среде env и возвращает
    список "net_worth" по всем шагам.
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
    # --------------------------------------------------
    # 1) Пути к папкам/файлам
    # --------------------------------------------------
    BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_folder = os.path.join(BASE_DIR, "data", "djia")   # папка с 30 CSV-файлами Dow
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # --------------------------------------------------
    # 2) Разбиваем весь DJIA-датасет на train/val/test
    # --------------------------------------------------
    train_df, val_df, test_df, tickers = make_portfolio_env_splits(data_folder)

    # --------------------------------------------------
    # 3) Инициализируем три окружения PortfolioEnv
    #    (передаём initial_balance=100_000)
    # --------------------------------------------------
    train_env = PortfolioEnv(df=train_df, initial_balance=100_000)
    val_env   = PortfolioEnv(df=val_df,   initial_balance=100_000)
    test_env  = PortfolioEnv(df=test_df,  initial_balance=100_000)

    # --------------------------------------------------
    # 4) Структура нейросети (policy_kwargs) — две скрытые MLP-слоя по 128
    #    **ВАЖНО**: здесь именно словарь, а не список-обёртка
    # --------------------------------------------------
    policy_kwargs = dict(
        net_arch = dict(pi=[128, 128], qf=[128, 128])
    )

    # --------------------------------------------------
    # 5) Гиперпараметры DDPG (точно как в статье Hongyang/Liu и из ВКР)
    # --------------------------------------------------
    # 5.1) buffer_size = 200_000
    # 5.2) batch_size = 64
    # 5.3) learning_rate = 1e-3
    # 5.4) τ = 0.005
    # 5.5) γ = 0.99
    # 5.6) train_freq = 1, gradient_steps = 1
    # 5.7) action_noise σ = 0.1
    # 5.8) policy_kwargs с net_arch = {"pi":[128,128], "qf":[128,128]}
    # 5.9) verbose = 1, tensorboard_log → для логов
    #
    n_actions = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    ddpg_kwargs = {
        "buffer_size": 200_000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "action_noise": action_noise,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "tensorboard_log": os.path.join(BASE_DIR, "logs", "ddpg_improved_portfolio")
    }

    # --------------------------------------------------
    # 6) Обучаем DDPG на train_env (2010–2018)
    # --------------------------------------------------
    print(">>> Начинаем обучение Improved DDPG (Train портфель) …")
    model = DDPG("MlpPolicy", train_env, **ddpg_kwargs)
    model.learn(total_timesteps=400_000)
    model_path = os.path.join(results_dir, "ddpg_improved_portfolio.zip")
    model.save(model_path)
    print(">>> Модель сохранена в", model_path)

    # --------------------------------------------------
    # 7) Оценка модели на Validation (2019–2020)
    # --------------------------------------------------
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

    eq_val_df = pd.DataFrame({
        "step": np.arange(len(equity_val)),
        "net_worth": equity_val
    })
    eq_val_df.to_csv(os.path.join(results_dir, "equity_val_ddpg_portfolio.csv"), index=False)
    print(">>> Equity-кривая Validation сохранена в results/equity_val_ddpg_portfolio.csv")

    # --------------------------------------------------
    # 8) Оценка модели на Test (2021–2024)
    # --------------------------------------------------
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
    eq_test_df.to_csv(os.path.join(results_dir, "equity_test_ddpg_portfolio.csv"), index=False)
    print(">>> Equity-кривая Test сохранена в results/equity_test_ddpg_portfolio.csv")
