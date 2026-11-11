import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 1. 参数设置
# ============================================================

DATA_START = "2007-01-01"

# 滚动十年回测
FIRST_START_YEAR = 2010
LAST_START_YEAR = 2016
WINDOW_END_OFFSET = 10

SMA_LONG = 200

INITIAL_CAPITAL = 100000

OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["QQQ", "TQQQ", "QLD", "SGOV", "SHV"]

STRATEGY_NAMES = [
    "极简趋势策略(斜率过滤)",
    "TQQQ_SMA200",
    "QQQ_SMA200",
    "QQQ买入持有"
]


# ============================================================
# 2. 下载历史行情
# ============================================================

def normalize_index(index):
    index = pd.to_datetime(index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def extract_price_column(df, field):
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)
            if field in values:
                result = df.xs(field, axis=1, level=level)
                if isinstance(result, pd.DataFrame):
                    return result.iloc[:, 0]
                return result
        raise ValueError(f"无法找到行情字段：{field}")
    if field not in df.columns:
        raise ValueError(f"无法找到行情字段：{field}")
    return df[field]


def download_one(ticker):
    print(f"正在下载 {ticker} ...")
    df = yf.download(ticker, start=DATA_START, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"{ticker} 行情数据为空，请检查数据源")
    
    open_price = extract_price_column(df, "Open")
    close_price = extract_price_column(df, "Close")
    
    result = pd.DataFrame({
        f"{ticker}_Open": open_price,
        f"{ticker}_Close": close_price
    })
    result.index = normalize_index(result.index)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def download_all_data():
    frames = [download_one(ticker) for ticker in TICKERS]
    data = pd.concat(frames, axis=1).sort_index()
    data = data.dropna(subset=["QQQ_Open", "QQQ_Close"])
    if data.empty:
        raise RuntimeError("没有获取到有效的 QQQ 行情数据")
    return data


# ============================================================
# 3. 计算均线及核心条件
# ============================================================

def calculate_indicators(data):
    df = data.copy()

    df["SMA200"] = df["QQQ_Close"].rolling(SMA_LONG, min_periods=SMA_LONG).mean()

    above_200 = df["QQQ_Close"] > df["SMA200"]
    below_200 = df["QQQ_Close"] < df["SMA200"]

    df["above200_2d"] = above_200 & above_200.shift(1).fillna(False).astype(bool)
    df["below200_2d"] = below_200 & below_200.shift(1).fillna(False).astype(bool)

    # 核心升级：计算 SMA200 的斜率 (今天比昨天高，说明趋势向上)
    df["SMA200_Slope_Up"] = df["SMA200"] > df["SMA200"].shift(1)

    return df


# ============================================================
# 4. 极简趋势策略 (斜率过滤)
# ============================================================

def generate_minimal_slope_signals(df):
    signals = []
    current_position = "DEFENSIVE"

    for _, row in df.iterrows():
        if pd.isna(row["SMA200"]):
            signals.append(current_position)
            continue

        # 连续跌破 200 日线：彻底避险
        if row["below200_2d"]:
            current_position = "DEFENSIVE"
        
        # 连续站上 200 日线
        elif row["above200_2d"]:
            # 如果 200 日线斜率向上：主升浪，满仓 TQQQ
            if row["SMA200_Slope_Up"]:
                current_position = "TQQQ"
            # 如果 200 日线斜率向下：弱势反弹，只买 QLD
            else:
                current_position = "QLD"
                
        signals.append(current_position)

    return pd.Series(signals, index=df.index, name="Minimal_Slope")


# ============================================================
# 5. 对照组策略
# ============================================================

def generate_qqq_sma200_signals(df):
    signals = []
    current_position = "DEFENSIVE"
    for _, row in df.iterrows():
        if pd.isna(row["SMA200"]):
            signals.append(current_position)
            continue
        if row["below200_2d"]:
            current_position = "DEFENSIVE"
        elif row["above200_2d"]:
            current_position = "QQQ"
        signals.append(current_position)
    return pd.Series(signals, index=df.index, name="QQQ_SMA200")


def generate_tqqq_sma200_signals(df):
    signals = []
    current_position = "DEFENSIVE"
    for _, row in df.iterrows():
        if pd.isna(row["SMA200"]):
            signals.append(current_position)
            continue
        if row["below200_2d"]:
            current_position = "DEFENSIVE"
        elif row["above200_2d"]:
            current_position = "TQQQ"
        signals.append(current_position)
    return pd.Series(signals, index=df.index, name="TQQQ_SMA200")


# ============================================================
# 6. 资产解析与收益计算
# ============================================================

def get_defensive_ticker(date, data):
    if pd.notna(data.at[date, "SGOV_Open"]):
        return "SGOV"
    if pd.notna(data.at[date, "SHV_Open"]):
        return "SHV"
    return "CASH"


def resolve_positions(signals, data):
    positions = []
    first_tqqq_date = data["TQQQ_Open"].first_valid_index()

    for date, signal in signals.items():
        if signal == "DEFENSIVE":
            ticker = get_defensive_ticker(date, data)
        elif signal == "TQQQ":
            if first_tqqq_date is None or date < first_tqqq_date:
                ticker = "QLD"
            else:
                ticker = "TQQQ"
        else:
            ticker = signal
        
        # 如果目标资产缺数据（如早期 QLD 没有），允许 TQQQ 降级为 QLD
        if ticker != "CASH" and pd.isna(data.at[date, f"{ticker}_Open"]):
            if ticker == "TQQQ":
                ticker = "QLD"
            if pd.isna(data.at[date, f"{ticker}_Open"]):
                raise RuntimeError(f"{date} 目标资产 {ticker} 没有有效开盘价")

        positions.append(ticker)
    return pd.Series(positions, index=signals.index)


def calculate_asset_returns(data):
    returns = pd.DataFrame(index=data.index)
    for ticker in TICKERS:
        open_col = f"{ticker}_Open"
        returns[ticker] = (data[open_col].shift(-1) / data[open_col] - 1)
    returns["CASH"] = 0.0
    return returns


def calculate_strategy_returns(position, asset_returns):
    executed_position = position.shift(1)
    result = pd.Series(index=position.index, dtype=float)
    for date in position.index:
        ticker = executed_position.loc[date]
        if pd.isna(ticker):
            continue
        result.loc[date] = asset_returns.at[date, ticker]
    return result


# ============================================================
# 7. 绩效统计与回测主循环
# ============================================================

def select_window_returns(returns, start, end):
    next_dates = pd.Series(returns.index, index=returns.index).shift(-1)
    mask = (returns.index >= start) & (next_dates <= end)
    return returns.loc[mask].dropna()


def calculate_metrics(returns, position, start, end):
    selected = select_window_returns(returns, start, end)
    if len(selected) < 2:
        return None

    cumulative = (1 + selected).cumprod()
    equity_values = np.concatenate([[1.0], cumulative.values])
    equity = pd.Series(equity_values)
    total_return = equity.iloc[-1] - 1

    next_dates = pd.Series(returns.index, index=returns.index).shift(-1)
    actual_end_date = next_dates.loc[selected.index[-1]]
    elapsed_days = (actual_end_date - selected.index[0]).days
    years = elapsed_days / 365.25

    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1)
    max_drawdown = drawdown.min()

    daily_std = selected.std()
    volatility = daily_std * np.sqrt(252)
    sharpe = (selected.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else np.nan
    win_rate = (selected > 0).mean()

    executed = position.shift(1).reindex(selected.index).dropna()
    trades = executed.ne(executed.shift(1)).iloc[1:].sum() if len(executed) > 1 else 0

    return {
        "累计收益率": total_return,
        "年化收益率": cagr,
        "最大回撤": max_drawdown,
        "年化波动率": volatility,
        "夏普比率": sharpe,
        "日胜率": win_rate,
        "换仓次数": int(trades),
        "期末净值": equity.iloc[-1],
        "实际回测结束日": actual_end_date.strftime("%Y-%m-%d")
    }


def run_backtest(data):
    df = calculate_indicators(data)
    
    signals = {
        "极简趋势策略(斜率过滤)": generate_minimal_slope_signals(df),
        "QQQ_SMA200": generate_qqq_sma200_signals(df),
        "TQQQ_SMA200": generate_tqqq_sma200_signals(df)
    }
    
    positions = {name: resolve_positions(signal, df) for name, signal in signals.items()}
    positions["QQQ买入持有"] = pd.Series("QQQ", index=df.index)

    asset_returns = calculate_asset_returns(df)
    strategy_returns = {name: calculate_strategy_returns(positions[name], asset_returns) for name in positions}

    results = []
    window_returns = {}

    for start_year in range(FIRST_START_YEAR, LAST_START_YEAR + 1):
        end_year = start_year + WINDOW_END_OFFSET
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = min(pd.Timestamp(f"{end_year}-12-31"), df.index.max())

        if start_date >= end_date:
            continue

        window_name = f"{start_year}-{end_year}"
        print(f"\n正在回测：{window_name}")

        for strategy_name, returns in strategy_returns.items():
            metrics = calculate_metrics(returns, positions[strategy_name], start_date, end_date)
            if metrics is None:
                continue
            results.append({"回测区间": window_name, "策略": strategy_name, **metrics})
            
            window_returns[(window_name, strategy_name)] = select_window_returns(returns, start_date, end_date)

    return pd.DataFrame(results), window_returns, positions


# ============================================================
# 8. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("QQQ 极简趋势策略 滚动十年回测 (严厉审查版)")
    print("=" * 60)

    data = download_all_data()
    results, window_returns, positions = run_backtest(data)

    # 保存结果
    results.to_csv(OUTPUT_DIR / "十年回测绩效对比.csv", index=False, encoding="utf-8-sig")

    # 输出展示
    display_columns = ["回测区间", "策略", "累计收益率", "年化收益率", "最大回撤", "夏普比率", "换仓次数"]
    print("\n回测绩效对比：")
    print(results[display_columns].to_string(
        index=False,
        formatters={
            "累计收益率": "{:.2%}".format,
            "年化收益率": "{:.2%}".format,
            "最大回撤": "{:.2%}".format,
            "夏普比率": "{:.2f}".format
        }
    ))
    
    # 保存极简策略的交易记录，方便你审计
    executed = positions["极简趋势策略(斜率过滤)"].shift(1).dropna()
    changes = executed[executed.ne(executed.shift(1))]
    records = pd.DataFrame({"交易日期": changes.index, "持仓标的": changes.values})
    records.to_csv(OUTPUT_DIR / "极简策略_交易记录.csv", index=False, encoding="utf-8-sig")

    print(f"\n回测完成！结果已保存至 {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
