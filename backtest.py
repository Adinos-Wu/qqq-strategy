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
# 2. 下载历史行情（彻底重写，防弹版）
# ============================================================
def download_all_data():
    print("开始从 Yahoo Finance 下载数据...")
    data_dict = {}
    
    for ticker in TICKERS:
        print(f" -> 正在下载 {ticker} 数据...")
        # 使用最底层的参数，强行拉取单只股票
        df = yf.download(ticker, start=DATA_START, auto_adjust=True, progress=False)
        
        if df.empty:
            print(f" ⚠️ 警告: {ticker} 下载为空，使用空数据替代。")
            data_dict[f"{ticker}_Open"] = pd.Series(dtype=float)
            data_dict[f"{ticker}_Close"] = pd.Series(dtype=float)
            continue
            
        # 处理可能的多级列名（兼容新版 yfinance）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        data_dict[f"{ticker}_Open"] = df["Open"]
        data_dict[f"{ticker}_Close"] = df["Close"]
        
        print(f" ✅ {ticker} 下载完成，共 {len(df)} 条记录。")

    print("正在合并所有数据...")
    data = pd.DataFrame(data_dict)
    
    # 统一时区并去除重影
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    
    # 剔除 QQQ 缺失数据的日期
    data = data.dropna(subset=["QQQ_Open", "QQQ_Close"])
    print(f"数据合并完成，最终有效交易日：{len(data)} 天。\n")
    
    # 立刻保存原始行情，防止后面代码报错你看不到
    data.to_csv(OUTPUT_DIR / "1_历史行情原始数据.csv", encoding="utf-8-sig")
    print(f"✅ 原始行情数据已保存至: {OUTPUT_DIR / '1_历史行情原始数据.csv'}")
    
    return data

# ============================================================
# 3. 计算指标
# ============================================================
def calculate_indicators(data):
    df = data.copy()
    df["SMA200"] = df["QQQ_Close"].rolling(SMA_LONG, min_periods=SMA_LONG).mean()
    
    above_200 = df["QQQ_Close"] > df["SMA200"]
    below_200 = df["QQQ_Close"] < df["SMA200"]
    
    df["above200_2d"] = above_200 & above_200.shift(1).fillna(False).astype(bool)
    df["below200_2d"] = below_200 & below_200.shift(1).fillna(False).astype(bool)
    
    # 斜率：今天的均线值 > 昨天的均线值
    df["SMA200_Slope_Up"] = df["SMA200"] > df["SMA200"].shift(1)
    return df

# ============================================================
# 4. 四大状态生成信号
# ============================================================
def generate_pure_slope_signals(df):
    signals = []
    current_position = "DEFENSIVE"

    for _, row in df.iterrows():
        if pd.isna(row["SMA200"]):
            signals.append(current_position)
            continue

        # 状态一：跌破
        if row["below200_2d"]:
            current_position = "DEFENSIVE"
        
        # 状态二 / 状态三：站上
        elif row["above200_2d"]:
            if row["SMA200_Slope_Up"]:
                current_position = "TQQQ"
            else:
                current_position = "QLD"
                
        signals.append(current_position)

    return pd.Series(signals, index=df.index, name="Pure_Slope")

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
# 5. 资产解析与收益
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
    executed_position = position.shift(1)
    result = pd.Series(index=position.index, dtype=float)
    for date in position.index:
        ticker = executed_position.loc[date]
        if pd.notna(ticker):
            result.loc[date] = asset_returns.at[date, ticker]
    return result

# ============================================================
# 6. 绩效指标与主循环
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

def run_backtest(data):
    df = calculate_indicators(data)
    
    signals = {
        "纯斜率策略": generate_pure_slope_signals(df),
        "TQQQ_SMA200": generate_tqqq_sma200_signals(df)
    }
    
    positions = {name: resolve_positions(sig, data) for name, sig in signals.items()}
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

    return pd.DataFrame(results), positions

# ============================================================
# 7. 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("纯斜率四状态策略 回测（彻底修复下载版）")
    print("=" * 60)

    # 第一步：下载数据（会打印进度）
    raw_data = download_all_data()

    # 第二步：执行回测
    print("开始执行策略回测计算...")
    results_df, positions = run_backtest(raw_data)

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
    
    # 第三步：保存所有结果
    results_df.to_csv(OUTPUT_DIR / "2_纯斜率策略_回测结果.csv", index=False, encoding="utf-8-sig")

    executed = positions["纯斜率策略"].shift(1).dropna()
    changes = executed[executed.ne(executed.shift(1))]
    changes.to_csv(OUTPUT_DIR / "3_纯斜率策略_交易记录.csv", encoding="utf-8-sig")
    
    print(f"\n✅ 所有结果均已保存至文件夹：{OUTPUT_DIR.resolve()}")
    print("文件夹内应包含：原始数据、回测绩效、交易记录。")
