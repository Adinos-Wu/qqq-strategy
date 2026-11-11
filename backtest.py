import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# 1. 参数设置
# ============================================================
DATA_START = "2007-01-01"
FIRST_START_YEAR = 2010
LAST_START_YEAR = 2016
WINDOW_END_OFFSET = 10

SMA_LONG = 200
INITIAL_CAPITAL = 100000

OUTPUT_DIR = Path("backtest_results_pure_slope")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["QQQ", "TQQQ", "QLD", "SGOV", "SHV"]

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
        raise RuntimeError(f"{ticker} 行情数据为空")
    open_price = extract_price_column(df, "Open")
    close_price = extract_price_column(df, "Close")
    result = pd.DataFrame({f"{ticker}_Open": open_price, f"{ticker}_Close": close_price})
    result.index = normalize_index(result.index)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result

def download_all_data():
    frames = [download_one(ticker) for ticker in TICKERS]
    data = pd.concat(frames, axis=1).sort_index()
    data = data.dropna(subset=["QQQ_Open", "QQQ_Close"])
    return data

# ============================================================
# 3. 计算指标（仅用 SMA200 及斜率）
# ============================================================
def calculate_indicators(data):
    df = data.copy()
    df["SMA200"] = df["QQQ_Close"].rolling(SMA_LONG, min_periods=SMA_LONG).mean()
    
    # 连续2日站上/跌破
    above_200 = df["QQQ_Close"] > df["SMA200"]
    below_200 = df["QQQ_Close"] < df["SMA200"]
    df["above200_2d"] = above_200 & above_200.shift(1).fillna(False).astype(bool)
    df["below200_2d"] = below_200 & below_200.shift(1).fillna(False).astype(bool)
    
    # SMA200 斜率（今天大于昨天为向上）
    df["SMA200_Slope_Up"] = df["SMA200"] > df["SMA200"].shift(1)
    return df

# ============================================================
# 4. 严格按你的四大状态生成信号
# ============================================================
def generate_pure_slope_signals(df):
    signals = []
    current_position = "DEFENSIVE"

    for _, row in df.iterrows():
        if pd.isna(row["SMA200"]):
            signals.append(current_position)
            continue

        # 【状态一】连续2日跌破 200 日线 -> 彻底避险
        if row["below200_2d"]:
            current_position = "DEFENSIVE"
        
        # 【状态二 & 状态三】连续2日站上 200 日线
        elif row["above200_2d"]:
            # 如果均线斜率向上 -> 【状态三】主升趋势，满仓 TQQQ
            if row["SMA200_Slope_Up"]:
                current_position = "TQQQ"
            # 如果均线斜率向下 -> 【状态二】弱势反弹，持有 QLD
            else:
                current_position = "QLD"
        
        # 其他情况（价格在均线附近震荡，未满足连续2日条件）维持原仓位
        signals.append(current_position)

    return pd.Series(signals, index=df.index, name="Pure_Slope")

# ============================================================
# 5. 对照组策略
# ============================================================
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
    for date, signal in signals.items():
        if signal == "DEFENSIVE":
            ticker = get_defensive_ticker(date, data)
        else:
            ticker = signal
        
        if ticker != "CASH" and pd.isna(data.at[date, f"{ticker}_Open"]):
            if ticker == "TQQQ": ticker = "QLD"
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
    # T日收盘出信号，T+1日开盘执行
    executed_position = position.shift(1)
    result = pd.Series(index=position.index, dtype=float)
    for date in position.index:
        ticker = executed_position.loc[date]
        if pd.notna(ticker):
            result.loc[date] = asset_returns.at[date, ticker]
    return result

# ============================================================
# 7. 绩效指标与回测主循环
# ============================================================
def calculate_metrics(returns, position, start, end):
    next_dates = pd.Series(returns.index, index=returns.index).shift(-1)
    mask = (returns.index >= start) & (next_dates <= end)
    selected = returns.loc[mask].dropna()

    if len(selected) < 2:
        return None

    cumulative = (1 + selected).cumprod()
    equity = pd.Series(np.concatenate([[1.0], cumulative.values]))
    total_return = equity.iloc[-1] - 1

    actual_end_date = next_dates.loc[selected.index[-1]]
    years = (actual_end_date - selected.index[0]).days / 365.25
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    max_drawdown = (equity / equity.cummax() - 1).min()

    daily_std = selected.std()
    sharpe = (selected.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else np.nan

    executed = position.shift(1).reindex(selected.index).dropna()
    trades = executed.ne(executed.shift(1)).iloc[1:].sum() if len(executed) > 1 else 0

    return {
        "累计收益率": total_return,
        "年化收益率": cagr,
        "最大回撤": max_drawdown,
        "夏普比率": sharpe,
        "换仓次数": int(trades),
        "实际回测结束日": actual_end_date.strftime("%Y-%m-%d")
    }

def run_backtest():
    data = download_all_data()
    df = calculate_indicators(data)
    
    # 生成信号
    signals = {
        "纯斜率策略": generate_pure_slope_signals(df),
        "TQQQ_SMA200": generate_tqqq_sma200_signals(df)
    }
    
    # 持仓解析
    positions = {name: resolve_positions(sig, data) for name, sig in signals.items()}
    
    # 收益计算
    asset_returns = calculate_asset_returns(data)
    strategy_returns = {name: calculate_strategy_returns(positions[name], asset_returns) for name in positions}

    results = []
    for start_year in range(FIRST_START_YEAR, LAST_START_YEAR + 1):
        end_year = start_year + WINDOW_END_OFFSET
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = min(pd.Timestamp(f"{end_year}-12-31"), df.index.max())
        
        if start_date >= end_date: continue
        
        window_name = f"{start_year}-{end_year}"
        for strategy_name, returns in strategy_returns.items():
            metrics = calculate_metrics(returns, positions[strategy_name], start_date, end_date)
            if metrics:
                results.append({"回测区间": window_name, "策略": strategy_name, **metrics})

    return pd.DataFrame(results), positions, strategy_returns

# ============================================================
# 8. 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("纯斜率四状态策略 回测")
    print("=" * 60)

    results_df, positions, _ = run_backtest()

    print("\n回测绩效对比：")
    print(results_df.to_string(
        index=False,
        formatters={
            "累计收益率": "{:.2%}".format,
            "年化收益率": "{:.2%}".format,
            "最大回撤": "{:.2%}".format,
            "夏普比率": "{:.2f}".format
        }
    ))
    
    results_df.to_csv(OUTPUT_DIR / "纯斜率策略_回测结果.csv", index=False, encoding="utf-8-sig")

    # 导出交易记录
    executed = positions["纯斜率策略"].shift(1).dropna()
    changes = executed[executed.ne(executed.shift(1))]
    changes.to_csv(OUTPUT_DIR / "纯斜率策略_交易记录.csv", encoding="utf-8-sig")
    print(f"\n结果已保存至：{OUTPUT_DIR.resolve()}")
