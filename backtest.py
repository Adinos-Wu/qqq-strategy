
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# 1. 参数设置
# ============================================================

DATA_START = "2007-01-01"

# 回测年份：2010-2020、2011-2021……2016-2026
FIRST_START_YEAR = 2010
LAST_START_YEAR = 2016
WINDOW_END_OFFSET = 10

SMA_SHORT = 120
SMA_LONG = 200

INITIAL_CAPITAL = 100000

OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TICKERS = ["QQQ", "TQQQ", "QLD", "SGOV", "SHV"]

# ============================================================
# 2. 下载历史数据
# ============================================================

def extract_price_column(df, field):
    """兼容 yfinance 普通列和 MultiIndex 列。"""

    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)

            if field in values:
                result = df.xs(
                    field,
                    axis=1,
                    level=level
                )

                if isinstance(result, pd.DataFrame):
                    return result.iloc[:, 0]

                return result

        raise ValueError(f"无法找到行情字段：{field}")

    if field not in df.columns:
        raise ValueError(f"无法找到行情字段：{field}")

    return df[field]


def normalize_index(index):
    index = pd.to_datetime(index)

    if index.tz is not None:
        index = index.tz_localize(None)

    return index.normalize()


def download_one(ticker):

    print(f"正在下载 {ticker} ...")

    df = yf.download(
        ticker,
        start=DATA_START,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        raise RuntimeError(f"{ticker} 行情数据为空")

    open_price = extract_price_column(df, "Open")
    close_price = extract_price_column(df, "Close")

    result = pd.DataFrame({
        f"{ticker}_Open": open_price,
        f"{ticker}_Close": close_price
    })

    result.index = normalize_index(result.index)

    result = result[
        ~result.index.duplicated(keep="last")
    ].sort_index()

    return result


def download_all_data():

    frames = []

    for ticker in TICKERS:
        frames.append(download_one(ticker))

    data = pd.concat(frames, axis=1).sort_index()

    # 以 QQQ 的交易日期为主
    data = data.dropna(subset=["QQQ_Close", "QQQ_Open"])

    return data


# ============================================================
# 3. 计算均线及连续交易日条件
# ============================================================

def calculate_indicators(data):

    df = data.copy()

    df["SMA120"] = df["QQQ_Close"].rolling(
        SMA_SHORT,
        min_periods=SMA_SHORT
    ).mean()

    df["SMA200"] = df["QQQ_Close"].rolling(
        SMA_LONG,
        min_periods=SMA_LONG
    ).mean()

    above_120 = df["QQQ_Close"] > df["SMA120"]
    below_120 = df["QQQ_Close"] < df["SMA120"]

    above_200 = df["QQQ_Close"] > df["SMA200"]
    below_200 = df["QQQ_Close"] < df["SMA200"]

    df["above120_2d"] = (
        above_120 & above_120.shift(1).fillna(False)
    )

    df["below120_2d"] = (
        below_120 & below_120.shift(1).fillna(False)
    )

    df["above200_2d"] = (
        above_200 & above_200.shift(1).fillna(False)
    )

    df["below200_2d"] = (
        below_200 & below_200.shift(1).fillna(False)
    )

    df["bullish_trend"] = df["SMA120"] > df["SMA200"]

    return df


# ============================================================
# 4. 四状态策略
# ============================================================

def generate_four_state_signals(df):

    signals = []

    # 初始没有有效信号时，先持有防御资产
    current_position = "DEFENSIVE"

    for _, row in df.iterrows():

        if pd.isna(row["SMA200"]):
            signals.append(current_position)
            continue

        # 状态1：避险
        if row["below200_2d"]:
            current_position = "DEFENSIVE"

        # 状态3：进攻
        elif (
            row["bullish_trend"]
            and row["above120_2d"]
        ):
            current_position = "TQQQ"

        # 状态4：降级
        elif row["below120_2d"]:
            current_position = "QLD"

        # 状态2：趋势恢复
        elif row["above200_2d"]:
            current_position = "QLD"

        # 其他情况维持原仓位

        signals.append(current_position)

    return pd.Series(
        signals,
        index=df.index,
        name="Four_State"
    )


# ============================================================
# 5. 两种均线对照策略
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

    return pd.Series(signals, index=df.index)


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

    return pd.Series(signals, index=df.index)


# ============================================================
# 6. 防御资产选择
# ============================================================

def get_defensive_ticker(date, data):

    # SGOV 上市前使用 SHV
    sgov_open = data.at[date, "SGOV_Open"]

    if pd.notna(sgov_open):
        return "SGOV"

    shv_open = data.at[date, "SHV_Open"]

    if pd.notna(shv_open):
        return "SHV"

    raise RuntimeError(f"{date} 没有可用的防御资产行情")


def resolve_positions(signals, data):

    positions = []

    for date, signal in signals.items():

        if signal == "DEFENSIVE":
            ticker = get_defensive_ticker(date, data)

        else:
            ticker = signal

            # 防止资产上市前出现无效持仓
            if pd.isna(data.at[date, f"{ticker}_Open"]):
                ticker = "QLD"

        positions.append(ticker)

    return pd.Series(positions, index=signals.index)


# ============================================================
# 7. 计算开盘到下一交易日开盘收益
# ============================================================

def calculate_asset_returns(data):

    returns = pd.DataFrame(index=data.index)

    for ticker in TICKERS:

        open_col = f"{ticker}_Open"

        # 日期 d 的收益：Open(d) 到 Open(下一交易日)
        returns[ticker] = (
            data[open_col].shift(-1) / data[open_col] - 1
        )

    return returns


def calculate_strategy_returns(position, asset_returns):

    # 信号在收盘后产生，下一交易日开盘执行
    # 因此日期 d 的收益使用前一交易日收盘产生的信号
    executed_position = position.shift(1)

    result = pd.Series(
        index=position.index,
        dtype=float
    )

    for date in position.index:

        ticker = executed_position.loc[date]

        if pd.isna(ticker):
            continue

        result.loc[date] = asset_returns.at[date, ticker]

    return result


# ============================================================
# 8. 绩效指标
# ============================================================

def calculate_metrics(returns, position, start, end):

    # 只统计区间内的有效收益
    # 每个日期的收益从该日开盘至下一交易日开盘
    selected = returns.loc[
        (returns.index >= start)
        & (returns.index <= end)
    ].dropna()

    if len(selected) < 2:
        return None

    equity = (1 + selected).cumprod()

    total_return = equity.iloc[-1] - 1

    elapsed_days = (
        selected.index[-1] - selected.index[0]
    ).days

    years = elapsed_days / 365.25

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0 else np.nan
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    max_drawdown = drawdown.min()

    volatility = selected.std() * np.sqrt(252)

    sharpe = (
        selected.mean() / selected.std() * np.sqrt(252)
        if selected.std() > 0 else np.nan
    )

    win_rate = (selected > 0).mean()

    executed = position.shift(1).loc[selected.index].dropna()

    trades = (
        executed.ne(executed.shift(1)).sum()
        if len(executed) > 1 else 0
    )

    return {
        "累计收益率": total_return,
        "年化收益率": cagr,
        "最大回撤": max_drawdown,
        "年化波动率": volatility,
        "夏普比率": sharpe,
        "日胜率": win_rate,
        "换仓次数": int(trades),
        "期末净值": equity.iloc[-1]
    }


# ============================================================
# 9. 年度收益
# ============================================================

def calculate_annual_returns(returns):

    annual = returns.dropna().groupby(
        returns.dropna().index.year
    ).apply(
        lambda x: (1 + x).prod() - 1
    )

    return annual


# ============================================================
# 10. 滚动十年回测
# ============================================================

def run_backtest(data):

    df = calculate_indicators(data)

    signals = {
        "四状态策略": generate_four_state_signals(df),
        "QQQ_SMA200": generate_qqq_sma200_signals(df),
        "TQQQ_SMA200": generate_tqqq_sma200_signals(df)
    }

    positions = {
        name: resolve_positions(signal, df)
        for name, signal in signals.items()
    }

    # QQQ 买入持有
    positions["QQQ买入持有"] = pd.Series(
        "QQQ",
        index=df.index
    )

    asset_returns = calculate_asset_returns(df)

    strategy_returns = {
        name: calculate_strategy_returns(
            positions[name],
            asset_returns
        )
        for name in positions
    }

    # 所有策略统一从同一交易日开始计算
    results = []
    annual_results = []
    window_returns = {}

    for start_year in range(
        FIRST_START_YEAR,
        LAST_START_YEAR + 1
    ):

        end_year = start_year + WINDOW_END_OFFSET

        start_date = pd.Timestamp(
            f"{start_year}-01-01"
        )

        # 包含结束年份内截至最新可用日期的数据
        end_date = min(
            pd.Timestamp(f"{end_year}-12-31"),
            df.index.max()
        )

        # 以完整交易区间为准
        if start_date >= end_date:
            continue

        window_name = f"{start_year}-{end_year}"

        print(f"\n正在回测：{window_name}")

        for strategy_name, returns in strategy_returns.items():

            metrics = calculate_metrics(
                returns,
                positions[strategy_name],
                start_date,
                end_date
            )

            if metrics is None:
                continue

            results.append({
                "回测区间": window_name,
                "策略": strategy_name,
                **metrics
            })

            selected = returns.loc[
                (returns.index >= start_date)
                & (returns.index <= end_date)
            ].dropna()

            annual = calculate_annual_returns(selected)

            for year, annual_return in annual.items():

                annual_results.append({
                    "回测区间": window_name,
                    "策略": strategy_name,
                    "年份": year,
                    "年度收益率": annual_return
                })

            window_returns[
                (window_name, strategy_name)
            ] = selected

    return (
        pd.DataFrame(results),
        pd.DataFrame(annual_results),
        window_returns,
        positions,
        strategy_returns
    )


# ============================================================
# 11. 绘制收益曲线
# ============================================================

def plot_window_curves(window_returns):

    for (window_name, strategy_name), returns in window_returns.items():

        equity = (1 + returns).cumprod()

        plt.figure(figsize=(12, 6))

        plt.plot(
            equity.index,
            equity.values,
            label=strategy_name
        )

        plt.title(
            f"{window_name} - {strategy_name}"
        )

        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(True)

        filename = (
            f"{window_name}_{strategy_name}.png"
            .replace("/", "_")
        )

        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# 12. 输出交易记录
# ============================================================

def save_trade_records(positions):

    for strategy_name, position in positions.items():

        executed = position.shift(1).dropna()

        changes = executed[
            executed.ne(executed.shift(1))
        ]

        records = pd.DataFrame({
            "交易日期": changes.index,
            "持仓标的": changes.values
        })

        records.to_csv(
            OUTPUT_DIR / f"{strategy_name}_交易记录.csv",
            index=False,
            encoding="utf-8-sig"
        )


# ============================================================
# 13. 主程序
# ============================================================

def main():

    print("=" * 60)
    print("QQQ 趋势策略滚动十年回测")
    print("=" * 60)

    data = download_all_data()

    data.to_csv(
        OUTPUT_DIR / "历史行情数据.csv",
        encoding="utf-8-sig"
    )

    print("\n行情数据下载完成")
    print(f"数据起始日：{data.index.min().date()}")
    print(f"数据截止日：{data.index.max().date()}")

    (
        results,
        annual_results,
        window_returns,
        positions,
        strategy_returns
    ) = run_backtest(data)

    # 输出绩效对比
    results.to_csv(
        OUTPUT_DIR / "十年回测绩效对比.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 输出年度收益
    annual_results.to_csv(
        OUTPUT_DIR / "年度收益对比.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 输出交易记录
    save_trade_records(positions)

    # 绘制每个区间的收益曲线
    plot_window_curves(window_returns)

    # 控制台输出
    if not results.empty:

        display_columns = [
            "回测区间",
            "策略",
            "累计收益率",
            "年化收益率",
            "最大回撤",
            "夏普比率",
            "换仓次数"
        ]

        print("\n回测绩效对比：")

        print(
            results[display_columns].to_string(
                index=False,
                formatters={
                    "累计收益率": "{:.2%}".format,
                    "年化收益率": "{:.2%}".format,
                    "最大回撤": "{:.2%}".format,
                    "夏普比率": "{:.2f}".format
                }
            )
        )

    print("\n回测完成！")
    print(f"结果保存位置：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
