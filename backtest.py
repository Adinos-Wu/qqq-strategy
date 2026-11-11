
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

# 滚动十年回测：
# 2010-2020、2011-2021……2016-2026
FIRST_START_YEAR = 2010
LAST_START_YEAR = 2016
WINDOW_END_OFFSET = 10

SMA_SHORT = 120
SMA_LONG = 200

INITIAL_CAPITAL = 100000

OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["QQQ", "TQQQ", "QLD", "SGOV", "SHV"]

STRATEGY_NAMES = [
    "四状态策略",
    "QQQ_SMA200",
    "TQQQ_SMA200",
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
    """
    兼容 yfinance 普通列和 MultiIndex 列。
    """

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

        raise ValueError(
            f"无法找到行情字段：{field}"
        )

    if field not in df.columns:
        raise ValueError(
            f"无法找到行情字段：{field}"
        )

    return df[field]


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
        raise RuntimeError(
            f"{ticker} 行情数据为空，请检查数据源"
        )

    open_price = extract_price_column(
        df,
        "Open"
    )

    close_price = extract_price_column(
        df,
        "Close"
    )

    result = pd.DataFrame({
        f"{ticker}_Open": open_price,
        f"{ticker}_Close": close_price
    })

    result.index = normalize_index(
        result.index
    )

    result = result[
        ~result.index.duplicated(keep="last")
    ].sort_index()

    return result


def download_all_data():

    frames = []

    for ticker in TICKERS:

        frames.append(
            download_one(ticker)
        )

    data = pd.concat(
        frames,
        axis=1
    ).sort_index()

    # 以 QQQ 的交易日期为主
    data = data.dropna(
        subset=[
            "QQQ_Open",
            "QQQ_Close"
        ]
    )

    if data.empty:
        raise RuntimeError(
            "没有获取到有效的 QQQ 行情数据"
        )

    return data


# ============================================================
# 3. 计算均线及连续交易日条件
# ============================================================

def calculate_indicators(data):

    df = data.copy()

    df["SMA120"] = (
        df["QQQ_Close"]
        .rolling(
            SMA_SHORT,
            min_periods=SMA_SHORT
        )
        .mean()
    )

    df["SMA200"] = (
        df["QQQ_Close"]
        .rolling(
            SMA_LONG,
            min_periods=SMA_LONG
        )
        .mean()
    )

    above_120 = (
        df["QQQ_Close"] > df["SMA120"]
    )

    below_120 = (
        df["QQQ_Close"] < df["SMA120"]
    )

    above_200 = (
        df["QQQ_Close"] > df["SMA200"]
    )

    below_200 = (
        df["QQQ_Close"] < df["SMA200"]
    )

    df["above120_2d"] = (
        above_120
        & above_120.shift(1).fillna(False).astype(bool)
    )

    df["below120_2d"] = (
        below_120
        & below_120.shift(1).fillna(False).astype(bool)
    )

    df["above200_2d"] = (
        above_200
        & above_200.shift(1).fillna(False).astype(bool)
    )

    df["below200_2d"] = (
        below_200
        & below_200.shift(1).fillna(False).astype(bool)
    )

    df["bullish_trend"] = (
        df["SMA120"] > df["SMA200"]
    )

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

    return pd.Series(
        signals,
        index=df.index,
        name="QQQ_SMA200"
    )


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

    return pd.Series(
        signals,
        index=df.index,
        name="TQQQ_SMA200"
    )


# ============================================================
# 6. 防御资产选择及实际持仓解析
# ============================================================

def get_defensive_ticker(date, data):

    # 优先使用 SGOV
    sgov_open = data.at[
        date,
        "SGOV_Open"
    ]

    if pd.notna(sgov_open):

        return "SGOV"

    # SGOV 不可用时使用 SHV
    shv_open = data.at[
        date,
        "SHV_Open"
    ]

    if pd.notna(shv_open):

        return "SHV"

    # SHV 尚未有历史行情时，暂时持有现金
    first_shv_date = (
        data["SHV_Open"].first_valid_index()
    )

    if (
        first_shv_date is None
        or date < first_shv_date
    ):

        return "CASH"

    # 已经进入 SHV 的历史交易区间，
    # 却没有行情，不允许静默跳过
    raise RuntimeError(
        f"{date} SGOV和SHV均无行情，请检查数据"
    )


def resolve_positions(signals, data):

    positions = []

    first_tqqq_date = (
        data["TQQQ_Open"].first_valid_index()
    )

    first_qld_date = (
        data["QLD_Open"].first_valid_index()
    )

    for date, signal in signals.items():

        if signal == "DEFENSIVE":

            ticker = get_defensive_ticker(
                date,
                data
            )

        elif signal == "TQQQ":

            # TQQQ 上市前使用 QLD 替代
            if (
                first_tqqq_date is None
                or date < first_tqqq_date
            ):

                ticker = "QLD"

            else:

                ticker = "TQQQ"

        else:

            ticker = signal

        if ticker != "CASH":

            open_price = data.at[
                date,
                f"{ticker}_Open"
            ]

            if pd.isna(open_price):

                # 仅允许 TQQQ 上市前替代 QLD
                if (
                    ticker == "TQQQ"
                    and (
                        first_tqqq_date is None
                        or date < first_tqqq_date
                    )
                ):

                    ticker = "QLD"

                    open_price = data.at[
                        date,
                        "QLD_Open"
                    ]

                if pd.isna(open_price):

                    raise RuntimeError(
                        f"{date} 目标资产 {ticker} "
                        f"没有有效开盘价"
                    )

        positions.append(ticker)

    return pd.Series(
        positions,
        index=signals.index
    )


# ============================================================
# 7. 计算开盘到下一交易日开盘收益
# ============================================================

def calculate_asset_returns(data):

    returns = pd.DataFrame(
        index=data.index
    )

    for ticker in TICKERS:

        open_col = f"{ticker}_Open"

        # 日期 d 的收益：
        # Open(d) 到 Open(下一交易日)
        returns[ticker] = (
            data[open_col].shift(-1)
            / data[open_col] - 1
        )

    # 现金仓位收益率设为 0
    returns["CASH"] = 0.0

    return returns


def calculate_strategy_returns(
    position,
    asset_returns
):

    # T日收盘产生信号，
    # T+1日开盘执行。
    #
    # 因此日期d的开盘至下一交易日开盘收益，
    # 使用前一交易日收盘产生的信号。

    executed_position = position.shift(1)

    result = pd.Series(
        index=position.index,
        dtype=float
    )

    for date in position.index:

        ticker = executed_position.loc[date]

        if pd.isna(ticker):

            continue

        result.loc[date] = (
            asset_returns.at[date, ticker]
        )

    return result


# ============================================================
# 8. 严格筛选回测区间
# ============================================================

def select_window_returns(
    returns,
    start,
    end
):

    # 收益索引代表收益开始日期
    # 下一交易日代表收益结束日期

    next_dates = pd.Series(
        returns.index,
        index=returns.index
    ).shift(-1)

    mask = (
        (returns.index >= start)
        & (next_dates <= end)
    )

    return returns.loc[mask].dropna()


# ============================================================
# 9. 绩效指标
# ============================================================

def calculate_metrics(
    returns,
    position,
    start,
    end
):

    selected = select_window_returns(
        returns,
        start,
        end
    )

    if len(selected) < 2:

        return None

    # 计算净值
    cumulative = (
        1 + selected
    ).cumprod()

    # 必须包含初始本金1.0，
    # 防止首日亏损没有进入最大回撤计算
    equity_values = np.concatenate([
        [1.0],
        cumulative.values
    ])

    equity = pd.Series(
        equity_values
    )

    total_return = (
        equity.iloc[-1] - 1
    )

    next_dates = pd.Series(
        returns.index,
        index=returns.index
    ).shift(-1)

    actual_end_date = (
        next_dates.loc[selected.index[-1]]
    )

    elapsed_days = (
        actual_end_date - selected.index[0]
    ).days

    years = elapsed_days / 365.25

    if years > 0:

        cagr = (
            equity.iloc[-1] ** (1 / years) - 1
        )

    else:

        cagr = np.nan

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    )

    max_drawdown = drawdown.min()

    daily_std = selected.std()

    volatility = (
        daily_std * np.sqrt(252)
    )

    # 夏普比率采用无风险利率为0的近似口径
    sharpe = (
        selected.mean() / daily_std * np.sqrt(252)
        if daily_std > 0
        else np.nan
    )

    win_rate = (
        selected > 0
    ).mean()

    executed = (
        position.shift(1)
        .reindex(selected.index)
        .dropna()
    )

    # 不把初始建仓计为一次换仓
    trades = (
        executed.ne(
            executed.shift(1)
        ).iloc[1:].sum()
        if len(executed) > 1
        else 0
    )

    return {

        "累计收益率": total_return,

        "年化收益率": cagr,

        "最大回撤": max_drawdown,

        "年化波动率": volatility,

        "夏普比率": sharpe,

        "日胜率": win_rate,

        "换仓次数": int(trades),

        "期末净值": equity.iloc[-1],

        "实际回测结束日": actual_end_date.strftime(
            "%Y-%m-%d"
        )
    }


# ============================================================
# 10. 年度收益
# ============================================================

def calculate_annual_returns(returns):

    valid = returns.dropna()

    if valid.empty:

        return pd.Series(dtype=float)

    # 按收益结束日期归属自然年度
    next_dates = pd.Series(
        returns.index,
        index=returns.index
    ).shift(-1)

    end_years = (
        next_dates.loc[valid.index].dt.year
    )

    annual = valid.groupby(
        end_years
    ).apply(
        lambda x: (
            (1 + x).prod() - 1
        )
    )

    return annual


# ============================================================
# 11. 滚动十年回测
# ============================================================

def run_backtest(data):

    df = calculate_indicators(data)

    signals = {

        "四状态策略":
            generate_four_state_signals(df),

        "QQQ_SMA200":
            generate_qqq_sma200_signals(df),

        "TQQQ_SMA200":
            generate_tqqq_sma200_signals(df)

    }

    positions = {

        name: resolve_positions(
            signal,
            df
        )

        for name, signal in signals.items()

    }

    # QQQ买入持有
    positions["QQQ买入持有"] = pd.Series(
        "QQQ",
        index=df.index
    )

    asset_returns = calculate_asset_returns(
        df
    )

    strategy_returns = {

        name: calculate_strategy_returns(
            positions[name],
            asset_returns
        )

        for name in positions

    }

    results = []

    annual_results = []

    window_returns = {}

    for start_year in range(
        FIRST_START_YEAR,
        LAST_START_YEAR + 1
    ):

        end_year = (
            start_year + WINDOW_END_OFFSET
        )

        start_date = pd.Timestamp(
            f"{start_year}-01-01"
        )

        # 结束日期不能超过最新可用行情
        end_date = min(
            pd.Timestamp(
                f"{end_year}-12-31"
            ),
            df.index.max()
        )

        if start_date >= end_date:

            continue

        window_name = (
            f"{start_year}-{end_year}"
        )

        print(
            f"\n正在回测：{window_name}"
        )

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

            selected = select_window_returns(
                returns,
                start_date,
                end_date
            )

            annual = calculate_annual_returns(
                selected
            )

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
# 12. 绘制收益曲线
# ============================================================

def plot_window_curves(window_returns):

    for (
        window_name,
        strategy_name
    ), returns in window_returns.items():

        equity = (
            1 + returns
        ).cumprod()

        # 从1开始显示净值
        equity = pd.concat([
            pd.Series(
                [1.0],
                index=[returns.index[0]]
            ),
            equity
        ])

        plt.figure(
            figsize=(12, 6)
        )

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
# 13. 输出交易记录
# ============================================================

def save_trade_records(positions):

    for strategy_name, position in positions.items():

        # 下一交易日实际执行的仓位
        executed = position.shift(1).dropna()

        if executed.empty:

            continue

        # 记录初始建仓及之后的仓位变化
        changes = executed[
            executed.ne(
                executed.shift(1)
            )
        ]

        records = pd.DataFrame({

            "交易日期": changes.index,

            "持仓标的": changes.values

        })

        records.to_csv(
            OUTPUT_DIR / (
                f"{strategy_name}_交易记录.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )


# ============================================================
# 14. 主程序
# ============================================================

def main():

    print("=" * 60)

    print(
        "QQQ 趋势策略滚动十年回测"
    )

    print("=" * 60)

    print(
        f"初始资金：${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"回测起始年份：{FIRST_START_YEAR}"
    )

    print(
        f"回测结束年份："
        f"{LAST_START_YEAR + WINDOW_END_OFFSET}"
    )

    print(
        f"均线参数：SMA{SMA_SHORT} / SMA{SMA_LONG}"
    )

    print("=" * 60)

    # 下载行情
    data = download_all_data()

    data.to_csv(
        OUTPUT_DIR / "历史行情数据.csv",
        encoding="utf-8-sig"
    )

    print("\n行情数据下载完成")

    print(
        f"数据起始日："
        f"{data.index.min().date()}"
    )

    print(
        f"数据截止日："
        f"{data.index.max().date()}"
    )

    # 检查各 ETF 首个有效交易日
    print("\n各 ETF 首个有效行情日期：")

    for ticker in TICKERS:

        first_date = (
            data[f"{ticker}_Open"].first_valid_index()
        )

        if first_date is None:

            print(
                f"{ticker}：无有效行情"
            )

        else:

            print(
                f"{ticker}：{first_date.date()}"
            )

    # 执行回测
    (
        results,
        annual_results,
        window_returns,
        positions,
        strategy_returns
    ) = run_backtest(data)

    if results.empty:

        raise RuntimeError(
            "没有生成有效回测结果，请检查数据和回测区间"
        )

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

    # 绘制收益曲线
    plot_window_curves(
        window_returns
    )

    # 控制台输出
    display_columns = [

        "回测区间",

        "策略",

        "累计收益率",

        "年化收益率",

        "最大回撤",

        "年化波动率",

        "夏普比率",

        "日胜率",

        "换仓次数",

        "期末净值"

    ]

    print("\n回测绩效对比：")

    print(
        results[display_columns].to_string(

            index=False,

            formatters={

                "累计收益率": "{:.2%}".format,

                "年化收益率": "{:.2%}".format,

                "最大回撤": "{:.2%}".format,

                "年化波动率": "{:.2%}".format,

                "夏普比率": "{:.2f}".format,

                "日胜率": "{:.2%}".format,

                "期末净值": "{:.4f}".format

            }

        )
    )

    print("\n回测完成！")

    print(
        f"结果保存位置：{OUTPUT_DIR.resolve()}"
    )

    print("\n生成的文件包括：")

    for file in sorted(OUTPUT_DIR.iterdir()):

        print(
            f" - {file.name}"
        )


if __name__ == "__main__":

    main()
