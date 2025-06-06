# src/sb3_examples/train_validate_ppo_portfolio.py

import os
import glob
import numpy as np
import pandas as pd
import random
import torch

from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from src.stock_env_portfolio import PortfolioEnv
from stable_baselines3 import PPO


def load_and_preprocess_djia(data_folder: str):
    """
    1) Считываем все CSV-файлы из data_folder/*.csv.
    2) Пропускаем первые три строки ("Price,…", "Ticker,…", "Date,,,,,") и задаём
       имена колонок ["Date","Close","High","Low","Open","Volume"].
    3) parse_dates=["Date"], index_col="Date" — чтобы Date стал индексом.
    4) Оставляем только ['Open','High','Low','Close','Volume'], переименовывая в
       '<TICKER>_Open', '<TICKER>_Close' и т.д.
    5) Inner-join по индексу (оставляем даты, общие для всех тикеров).
    6) Считаем техиндикаторы (MACD, RSI, CCI, ADX) для каждого тикера.
    7) Считаем Turbulence (60-дневное окно по доходностям Close).
    8) Dropna() и reset_index(), чтобы 'Date' вернулась в колонки.
    Возвращает: df_full и список tickers.
    """
    all_files = glob.glob(os.path.join(data_folder, "*.csv"))
    all_files.sort()

    list_df = []
    tickers = []

    for filepath in all_files:
        ticker = os.path.basename(filepath).replace(".csv", "")
        tickers.append(ticker)

        # 1) Читаем CSV, пропуская первые три служебные строки
        df_t = pd.read_csv(
            filepath,
            skiprows=3,
            names=["Date", "Close", "High", "Low", "Open", "Volume"],
            usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
            parse_dates=["Date"],
            index_col="Date"
        )

        # Проверяем обязательные колонки
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df_t.columns:
                raise ValueError(f"В файле {filepath} отсутствует столбец '{col}' после skiprows.")

        # 2) Переименовываем столбцы в формат '<TICKER>_Open', '<TICKER>_Close' и т.д.
        df_t = df_t.rename(columns={
            "Open":   f"{ticker}_Open",
            "High":   f"{ticker}_High",
            "Low":    f"{ticker}_Low",
            "Close":  f"{ticker}_Close",
            "Volume": f"{ticker}_Volume"
        })

        list_df.append(df_t)

    # 3) Inner-join по датам, общим для всех тикеров
    df_all = pd.concat(list_df, axis=1, join="inner").sort_index()

    # 4) Считаем тех. индикаторы для каждого тикера
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

    # 5) Объединяем цены и индикаторы
    df_full = pd.concat([df_all, df_indicators], axis=1)

    # 6) Считаем Turbulence (60-дневное окно по доходностям)
    close_cols = [f"{t}_Close" for t in tickers]
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

    # 7) Дропаем NaN (первые 60 строк + NA индикаторов) и reset_index()
    df_full = df_full.dropna().reset_index()  # => 'Date' снова обычный столбец

    return df_full, tickers


def make_portfolio_env_splits(df_full: pd.DataFrame):
    """
    Принимаем полный df_full (с колонками Date, ценами, индикаторами и Turbulence).
    Возвращаем словарь с ключами 'fold_{i}' → (train_df, val_df), где i = 1..4.
    Каждая пара формируется так (rolling‐window):

      fold_1: train 2010–2016, val 2017
      fold_2: train 2011–2017, val 2018
      fold_3: train 2012–2018, val 2019
      fold_4: train 2013–2019, val 2020

    После этого отдельно возвращаем «финальный» train 2010–2018 и val 2019–2020.
    """
    splits = {}

    # Убедимся, что колонка 'Date' есть и это datetime
    df_full["Date"] = pd.to_datetime(df_full["Date"])
    df_full = df_full.sort_values("Date").reset_index(drop=True)

    # 4 фолда CV
    years = [(2010, 2016, 2017),
             (2011, 2017, 2018),
             (2012, 2018, 2019),
             (2013, 2019, 2020)]

    for idx, (y_train_start, y_train_end, y_val) in enumerate(years, start=1):
        train_mask = (df_full["Date"] >= f"{y_train_start}-01-01") & (df_full["Date"] <= f"{y_train_end}-12-31")
        val_mask   = (df_full["Date"] >= f"{y_val}-01-01")         & (df_full["Date"] <= f"{y_val}-12-31")

        train_df = df_full.loc[train_mask].copy().reset_index(drop=True)
        val_df   = df_full.loc[val_mask].copy().reset_index(drop=True)

        splits[f"fold_{idx}"] = (train_df, val_df)

    # Финальный train/val (та, что была раньше):
    full_train_mask = (df_full["Date"] >= "2010-01-01") & (df_full["Date"] <= "2018-12-31")
    full_val_mask   = (df_full["Date"] >= "2019-01-01") & (df_full["Date"] <= "2020-12-31")
    train_full = df_full.loc[full_train_mask].copy().reset_index(drop=True)
    val_full   = df_full.loc[full_val_mask].copy().reset_index(drop=True)
    splits["final_train"] = (train_full, val_full)

    # Тестовый (2021–2024) отдельно положим, чтобы потом не искать заново
    test_mask = (df_full["Date"] >= "2021-01-01") & (df_full["Date"] <= "2024-12-31")
    test_df   = df_full.loc[test_mask].copy().reset_index(drop=True)
    splits["test"] = test_df

    return splits



def evaluate_on_env(env, model):
    """
    Прогоняет один полный эпизод на env и возвращает список net_worth на каждом шаге.
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
    # ----------------------------------
    # 1) Пути к папкам и файлам
    # ----------------------------------
    BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_folder = os.path.join(BASE_DIR, "data", "djia")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------------
    # 2) Загружаем и предобрабатываем все данные
    # ----------------------------------
    df_full, tickers = load_and_preprocess_djia(data_folder)

    # ----------------------------------
    # 3) Делаем кросс-валидационные разбиения
    # ----------------------------------
    splits = make_portfolio_env_splits(df_full)

    # ----------------------------------
    # 4) Гиперпараметры PPO (точно такие же, как в статье / дипломе)
    # ----------------------------------
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # две скрытых 128-нейрона
    )

    ppo_kwargs = dict(
        learning_rate   = 3e-4,    # как в статье / дипломе
        n_steps         = 2048,    # rollout_length
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 1e-4,    # из статьи / диплома
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = os.path.join(BASE_DIR, "logs", "ppo_portfolio")
    )

    total_timesteps = 300_000  # как и раньше

    # ----------------------------------
    # 5) Кросс-валидация (4 фолда)
    # ----------------------------------
    print("\n=== Начинаем k-fold (rolling-window) кросс-валидацию для PPO ===\n")
    metrics_cv = {}

    # Итерируемся только по ключам fold_1… fold_4
    for fold_name in ["fold_1", "fold_2", "fold_3", "fold_4"]:
        train_df, val_df = splits[fold_name]  # распакуем ровно кортеж (train_df, val_df)

        print(f"\n>>> {fold_name.upper()}: Train {train_df['Date'].min().date()}–{train_df['Date'].max().date()}, "
              f"Val {val_df['Date'].min().date()}–{val_df['Date'].max().date()}")

        # Создаём среды для этого фолда
        train_env = PortfolioEnv(df=train_df, initial_balance=1_000_000)
        val_env   = PortfolioEnv(df=val_df,   initial_balance=1_000_000)

        # Инициализируем новую модель PPO «с нуля» для каждого фолда
        model = PPO("MlpPolicy", train_env, **ppo_kwargs)
        model.learn(total_timesteps=total_timesteps)

        # Оцениваем на валидации данного фолда
        equity_val = evaluate_on_env(val_env, model)
        final_val   = equity_val[-1]
        roi_val     = (final_val - 1_000_000) / 1_000_000
        daily_rets  = np.diff(equity_val) / equity_val[:-1]
        sharpe_val  = 0.0
        if np.std(daily_rets) != 0:
            sharpe_val = (np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(252)

        peak = equity_val[0]
        max_dd_val = 0.0
        for x in equity_val:
            if x > peak:
                peak = x
            dd = (peak - x) / peak
            if dd > max_dd_val:
                max_dd_val = dd

        print(f"{fold_name}:  ROI={roi_val*100:.2f}%,  Sharpe={sharpe_val:.2f},  MaxDD={max_dd_val*100:.2f}%")

        metrics_cv[fold_name] = {
            "roi": roi_val,
            "sharpe": sharpe_val,
            "max_dd": max_dd_val
        }

    # Опционально: вывести сводную таблицу по всем 4 фолдам
    print("\n=== Результаты кросс-валидации по фолдам ===")
    df_cv_summary = pd.DataFrame(metrics_cv).T
    print(df_cv_summary, "\n")


    # ----------------------------------
    # 6) Финальное обучение на Full Train (2010–2018) и валидация на 2019–2020
    # ----------------------------------
    print("\n=== Финальное обучение на Train(2010–2018) и оценка на Val(2019–2020) ===")

    train_full, val_full = splits["final_train"]
    test_full = splits["test"]

    # Создаём среды
    train_env = PortfolioEnv(df=train_full, initial_balance=1_000_000)
    val_env   = PortfolioEnv(df=val_full,   initial_balance=1_000_000)
    test_env  = PortfolioEnv(df=test_full,  initial_balance=1_000_000)

    # Новая модель PPO на полном составе 2010–2018
    model = PPO("MlpPolicy", train_env, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(results_dir, "ppo_train_portfolio.zip")
    model.save(model_path)
    print(">>> Модель сохранена в", model_path)

    # Оценка на Validation (2019–2020)
    print(">>> Оцениваем на Validation …")
    equity_val = evaluate_on_env(val_env, model)
    final_net_val = equity_val[-1]

    roi_val = (final_net_val - 1_000_000) / 1_000_000
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

    print(f"Validation ROI:   {roi_val*100:.2f}%")
    print(f"Validation Sharpe: {sharpe_val:.2f}")
    print(f"Validation MaxDD:  {max_drawdown*100:.2f}%")
    print(f"Final net worth (Validation): {final_net_val:.2f}")

    eq_val_df = pd.DataFrame({
        "step": np.arange(len(equity_val)),
        "net_worth": equity_val
    })
    eq_val_df.to_csv(os.path.join(results_dir, "equity_val_ppo_portfolio.csv"), index=False)
    print(">>> Equity-кривая Validation сохранена в results/equity_val_ppo_portfolio.csv")

    # Оценка на Test (2021–2024)
    print(">>> Оцениваем на Test …")
    equity_test = evaluate_on_env(test_env, model)
    final_net_test = equity_test[-1]

    roi_test = (final_net_test - 1_000_000) / 1_000_000
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
    print(f"Final net worth (Test): {final_net_test:.2f}")

    eq_test_df = pd.DataFrame({
        "step": np.arange(len(equity_test)),
        "net_worth": equity_test
    })
    eq_test_df.to_csv(os.path.join(results_dir, "equity_test_ppo_portfolio.csv"), index=False)
    print(">>> Equity-кривая Test сохранена в results/equity_test_ppo_portfolio.csv")
