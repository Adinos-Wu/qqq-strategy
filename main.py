
import yfinance as yf
import pandas as pd
import requests
import os
import sys

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# 1. 基础配置
# ============================================================

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN")

# 是否推送无交易信号的日报
PUSH_NO_SIGNAL = True

PUSH_TIMEOUT = 15

# 上次成功推送的交易日期
STATE_FILE = Path(__file__).parent / "last_trade_date.txt"

# 上次成功推送后的策略目标持仓
POSITION_FILE = Path(__file__).parent / "strategy_position.txt"

# 策略参数
SMA_LONG = 200

# 美东收盘后预留15分钟，等待行情数据更新
MARKET_CLOSE_HOUR = 16
DATA_UPDATE_MINUTE = 15


# ============================================================
# 2. 读取上次推送日期
# ============================================================

def get_last_trade_date():

    if not STATE_FILE.exists():
        return None

    try:
        return STATE_FILE.read_text(
            encoding="utf-8"
        ).strip() or None

    except Exception as e:
        raise RuntimeError(
            f"读取交易日期记录失败：{e}"
        )


# ============================================================
# 3. 保存本次成功推送日期
# ============================================================

def save_last_trade_date(trade_date):

    STATE_FILE.write_text(
        trade_date,
        encoding="utf-8"
    )

    print(f"交易日期记录已更新：{trade_date}")


# ============================================================
# 4. 读取策略目标持仓
# ============================================================

def get_current_position():

    if not POSITION_FILE.exists():
        return None

    try:
        position = POSITION_FILE.read_text(
            encoding="utf-8"
        ).strip()

        if position in ("TQQQ", "SGOV"):
            return position

        return None

    except Exception as e:
        raise RuntimeError(
            f"读取策略持仓记录失败：{e}"
        )


# ============================================================
# 5. 保存策略目标持仓
# ============================================================

def save_current_position(position):

    if position not in ("TQQQ", "SGOV"):
        raise ValueError(
            f"无效的策略目标持仓：{position}"
        )

    POSITION_FILE.write_text(
        position,
        encoding="utf-8"
    )

    print(f"策略目标持仓已更新：{position}")


# ============================================================
# 6. 策略计算
# ============================================================

def run_strategy():

    try:

        # ----------------------------------------------------
        # 获取 QQQ 日线行情
        # ----------------------------------------------------

        qqq = yf.download(
            "QQQ",
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if qqq.empty:
            return "❌ 行情获取失败：QQQ 数据为空。"

        # 兼容 yfinance 多层列名
        if isinstance(qqq.columns, pd.MultiIndex):
            qqq.columns = qqq.columns.get_level_values(0)

        if "Close" not in qqq.columns:
            return "❌ 行情数据缺少 Close 列。"

        # 数据清理
        qqq = qqq.dropna(subset=["Close"])

        qqq = qqq[
            ~qqq.index.duplicated(keep="last")
        ]

        qqq = qqq.sort_index()

        # ----------------------------------------------------
        # 判断当天行情是否已经完成
        # ----------------------------------------------------

        now_et = datetime.now(
            ZoneInfo("America/New_York")
        )

        today_et = now_et.date()

        # 当天16:15之前，不使用当天日线
        if (
            now_et.hour < MARKET_CLOSE_HOUR
            or (
                now_et.hour == MARKET_CLOSE_HOUR
                and now_et.minute < DATA_UPDATE_MINUTE
            )
        ):

            qqq = qqq[
                pd.Index(qqq.index.date) < today_et
            ]

        if len(qqq) < SMA_LONG:
            return "❌ 有效行情数据不足，无法计算 SMA200。"

        # ----------------------------------------------------
        # 计算 SMA200
        # ----------------------------------------------------

        qqq["SMA200"] = qqq["Close"].rolling(
            window=SMA_LONG,
            min_periods=SMA_LONG
        ).mean()

        qqq = qqq.dropna(
            subset=["SMA200"]
        )

        if len(qqq) < 2:
            return "❌ 有效行情数据不足，无法计算两日确认信号。"

        # ----------------------------------------------------
        # 获取最近两个已完成交易日
        # ----------------------------------------------------

        t = qqq.iloc[-1]
        t1 = qqq.iloc[-2]

        close = float(t["Close"])
        sma200 = float(t["SMA200"])

        close_1 = float(t1["Close"])
        sma200_1 = float(t1["SMA200"])

        trade_date = qqq.index[-1].strftime("%Y-%m-%d")

        if close <= 0 or close_1 <= 0:
            return "❌ 收盘价异常，无法计算涨跌幅。"

        # ----------------------------------------------------
        # 今日行情信息
        # ----------------------------------------------------

        daily_change_price = close - close_1

        daily_change = (
            close / close_1 - 1
        ) * 100

        if daily_change > 0:
            daily_status = "上涨"

        elif daily_change < 0:
            daily_status = "下跌"

        else:
            daily_status = "持平"

        change_200 = (
            close / sma200 - 1
        ) * 100

        # ----------------------------------------------------
        # SMA200 两日确认
        # ----------------------------------------------------

        above_today = close > sma200
        below_today = close < sma200

        above_yesterday = close_1 > sma200_1
        below_yesterday = close_1 < sma200_1

        # 连续两个交易日收盘价高于 SMA200
        above_200_2d = (
            above_today and above_yesterday
        )

        # 连续两个交易日收盘价低于 SMA200
        below_200_2d = (
            below_today and below_yesterday
        )

        # ----------------------------------------------------
        # 读取当前策略目标持仓
        # ----------------------------------------------------

        current_position = get_current_position()

        # ----------------------------------------------------
        # 生成交易信号
        # ----------------------------------------------------

        target_position = current_position

        if above_200_2d:

            target_position = "TQQQ"

            if current_position != "TQQQ":

                signal = "🟢 两日确认：进攻信号"

                action = (
                    "买入 TQQQ，卖出原有持仓；"
                    "下一交易日开盘执行"
                )

            else:

                signal = "🟢 多头趋势延续"

                action = "继续持有 TQQQ，不需要换仓"

        elif below_200_2d:

            target_position = "SGOV"

            if current_position != "SGOV":

                signal = "🔴 两日确认：避险信号"

                action = (
                    "买入 SGOV，卖出原有持仓；"
                    "下一交易日开盘执行"
                )

            else:

                signal = "🔴 防御趋势延续"

                action = "继续持有 SGOV，不需要换仓"

        else:

            signal = "⚪ 尚未完成两日确认"

            if current_position:

                action = (
                    f"维持原策略目标持仓 {current_position}，"
                    "暂不调整"
                )

            else:

                action = (
                    "尚无已确认的策略目标持仓，"
                    "等待连续两日确认"
                )

        # ----------------------------------------------------
        # 判断价格位置
        # ----------------------------------------------------

        if above_today:
            position_status = "SMA200上方"

        elif below_today:
            position_status = "SMA200下方"

        else:
            position_status = "等于SMA200"

        # 两日确认状态
        if above_200_2d:
            confirmation_status = "连续2日站上SMA200"

        elif below_200_2d:
            confirmation_status = "连续2日跌破SMA200"

        elif above_today:
            confirmation_status = "今日站上，尚未完成两日确认"

        elif below_today:
            confirmation_status = "今日跌破，尚未完成两日确认"

        else:
            confirmation_status = "今日收盘价等于SMA200"

        # ----------------------------------------------------
        # 推送内容
        # ----------------------------------------------------

        current_position_text = (
            current_position
            if current_position
            else "尚未建立"
        )

        target_position_text = (
            target_position
            if target_position
            else "等待确认"
        )

        msg = (
            f"📊 QQQ SMA200 两日确认策略日报\n"
            f"📅 交易日：{trade_date}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💹 行情数据\n"
            f"💰 QQQ 收盘价：${close:.2f}\n"
            f"📊 今日涨跌：{daily_status} "
            f"${abs(daily_change_price):.2f} "
            f"（{daily_change:+.2f}%）\n"
            f"📈 SMA200：${sma200:.2f} "
            f"（偏离 {change_200:+.2f}%）\n"
            f"📍 当前价格位置：{position_status}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🔍 两日确认\n"
            f"昨日收盘：${close_1:.2f}\n"
            f"昨日 SMA200：${sma200_1:.2f}\n"
            f"确认状态：{confirmation_status}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🎯 交易信号\n"
            f"🚦 当前信号：{signal}\n"
            f"📦 原策略目标持仓：{current_position_text}\n"
            f"📦 新策略目标持仓：{target_position_text}\n"
            f"🔄 操作建议：{action}\n"
            f"⏰ 执行时间：下一交易日开盘\n"
        )

        return (
            msg,
            signal,
            trade_date,
            target_position
        )

    except Exception as e:

        return f"❌ 策略运行失败：{str(e)}"


# ============================================================
# 7. PushPlus 推送
# ============================================================

def pushplus_send(content):

    if not PUSHPLUS_TOKEN:
        raise RuntimeError(
            "未设置 PUSHPLUS_TOKEN，无法推送。"
        )

    url = "https://www.pushplus.plus/send"

    data = {
        "token": PUSHPLUS_TOKEN,
        "title": "QQQ SMA200 两日确认策略日报",
        "content": content,
        "template": "txt"
    }

    response = requests.post(
        url,
        json=data,
        timeout=PUSH_TIMEOUT
    )

    response.raise_for_status()

    try:
        result = response.json()

    except ValueError:
        raise RuntimeError(
            f"PushPlus 返回非 JSON 数据：{response.text}"
        )

    print("PushPlus 推送结果：", result)

    if str(result.get("code")) != "200":

        raise RuntimeError(
            f"PushPlus 推送失败：{result}"
        )

    return True


# ============================================================
# 8. 主程序
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("QQQ SMA200 两日确认策略")
    print("=" * 55)

    result = run_strategy()

    # 行情或策略计算失败
    if not isinstance(result, tuple):

        print(result)

        sys.exit(1)

    msg, signal, trade_date, target_position = result

    print(msg)

    # --------------------------------------------------------
    # 交易日期去重
    # --------------------------------------------------------

    last_trade_date = get_last_trade_date()

    if last_trade_date and trade_date <= last_trade_date:

        print(
            f"⏭️ 跳过推送：交易日期 {trade_date} "
            f"已经推送过，或早于最近推送日期 {last_trade_date}。"
        )

        sys.exit(0)

    # --------------------------------------------------------
    # 是否推送无交易信号日报
    # --------------------------------------------------------

    if (
        not PUSH_NO_SIGNAL
        and signal == "⚪ 尚未完成两日确认"
    ):

        print("当前未完成两日确认，按照配置跳过推送。")

        sys.exit(0)

    # --------------------------------------------------------
    # 推送成功后保存状态
    # --------------------------------------------------------

    pushplus_send(msg)

    # 只有推送成功后才更新策略目标持仓
    if target_position in ("TQQQ", "SGOV"):

        save_current_position(target_position)

    save_last_trade_date(trade_date)

    print("✅ 本次策略执行完成。")
