
import yfinance as yf
import pandas as pd
import requests
import os
import sys

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


# =========================
# 基础配置
# =========================

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN")

# 是否推送无交易信号的日报
PUSH_NO_SIGNAL = True

PUSH_TIMEOUT = 15

# 上次成功推送的交易日期记录
STATE_FILE = Path(__file__).parent / "last_trade_date.txt"


# =========================
# 读取上次推送日期
# =========================

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


# =========================
# 保存本次成功推送日期
# =========================

def save_last_trade_date(trade_date):

    STATE_FILE.write_text(
        trade_date,
        encoding="utf-8"
    )

    print(f"交易日期记录已更新：{trade_date}")


# =========================
# 策略计算
# =========================

def run_strategy():

    try:
        # 获取 QQQ 日线数据
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

        if len(qqq) < 200:
            return "❌ 有效行情数据不足，无法计算 SMA200。"

        # 美东时间
        now_et = datetime.now(
            ZoneInfo("America/New_York")
        )

        today_et = now_et.date()

        # 美股当天尚未完成收盘数据更新时，剔除当天日线
        if now_et.hour < 16 or (
            now_et.hour == 16 and now_et.minute < 15
        ):

            qqq = qqq[
                pd.Index(qqq.index.date) < today_et
            ]

        if len(qqq) < 200:
            return "❌ 有效行情数据不足，无法计算 SMA200。"

        # =========================
        # 计算均线
        # =========================

        qqq["SMA120"] = qqq["Close"].rolling(120).mean()

        qqq["SMA200"] = qqq["Close"].rolling(200).mean()

        qqq = qqq.dropna(
            subset=["SMA120", "SMA200"]
        )

        if len(qqq) < 2:
            return "❌ 有效行情数据不足，无法计算交易信号。"

        # 最近两个已完成交易日
        t = qqq.iloc[-1]
        t1 = qqq.iloc[-2]

        close = float(t["Close"])
        sma120 = float(t["SMA120"])
        sma200 = float(t["SMA200"])

        close_1 = float(t1["Close"])
        sma120_1 = float(t1["SMA120"])
        sma200_1 = float(t1["SMA200"])

        trade_date = qqq.index[-1].strftime("%Y-%m-%d")

        if close <= 0 or close_1 <= 0:
            return "❌ 收盘价异常，无法计算涨跌幅。"

        # =========================
        # 每日涨跌幅
        # =========================

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

        # =========================
        # 连续两日条件
        # =========================

        # 连续两日跌破 SMA200
        drop_200_2d = (
            close < sma200 and
            close_1 < sma200_1
        )

        # 连续两日站上 SMA200
        above_200_2d = (
            close > sma200 and
            close_1 > sma200_1
        )

        # 连续两日跌破 SMA120
        drop_120_2d = (
            close < sma120 and
            close_1 < sma120_1
        )

        # 连续两日站上 SMA120
        above_120_2d = (
            close > sma120 and
            close_1 > sma120_1
        )

        # 多头排列状态
        bullish_trend = sma120 > sma200

        # =========================
        # 交易信号
        # =========================

        if drop_200_2d:

            signal = "🔴 避险信号"

            action = "卖出 QLD / TQQQ，切换至 SGOV"

        elif drop_120_2d:

            signal = "🟡 降级信号"

            action = "卖出 TQQQ，切换至 QLD"

        elif above_200_2d:

            if bullish_trend and above_120_2d:

                signal = "🟢 进攻信号"

                action = "切换至 TQQQ"

            elif not bullish_trend:

                signal = "🟡 反弹参与信号"

                action = "切换至 QLD"

            else:

                signal = "🟡 趋势确认等待"

                action = "维持观察，暂不调整"

        elif above_120_2d and bullish_trend:

            signal = "🟢 重新进攻信号"

            action = "切换至 TQQQ"

        else:

            signal = "⚪ 暂无交易信号"

            action = "维持原有配置"

        # =========================
        # 行情信息
        # =========================

        change_120 = (
            close / sma120 - 1
        ) * 100

        change_200 = (
            close / sma200 - 1
        ) * 100

        position_200 = (
            "位于 SMA200 上方"
            if close > sma200
            else "位于 SMA200 下方"
        )

        position_120 = (
            "位于 SMA120 上方"
            if close > sma120
            else "位于 SMA120 下方"
        )

        cross_status = (
            "SMA120 > SMA200（多头排列）"
            if bullish_trend
            else "SMA120 < SMA200（空头排列）"
        )

        # =========================
        # 推送内容
        # =========================

        msg = (
            f"📊 QQQ 行情趋势策略日报\n"
            f"📅 交易日：{trade_date}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💹 行情数据\n"
            f"💰 QQQ 收盘价：${close:.2f}\n"
            f"📊 今日涨跌：{daily_status} "
            f"${abs(daily_change_price):.2f} "
            f"（{daily_change:+.2f}%）\n"
            f"📈 SMA120：${sma120:.2f}（{change_120:+.2f}%）\n"
            f"📉 SMA200：${sma200:.2f}（{change_200:+.2f}%）\n"
            f"🔀 均线关系：{cross_status}\n"
            f"📍 价格位置：{position_120}，{position_200}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🎯 交易信号\n"
            f"🚦 当前信号：{signal}\n"
            f"🔄 操作建议：{action}\n"
        )

        return msg, signal, trade_date

    except Exception as e:

        return f"❌ 策略运行失败：{str(e)}"


# =========================
# PushPlus 推送
# =========================

def pushplus_send(content):

    if not PUSHPLUS_TOKEN:
        raise RuntimeError(
            "未设置 PUSHPLUS_TOKEN，无法推送。"
        )

    url = "https://www.pushplus.plus/send"

    data = {
        "token": PUSHPLUS_TOKEN,
        "title": "QQQ 行情趋势策略日报",
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


# =========================
# 主程序
# =========================

if __name__ == "__main__":

    result = run_strategy()

    # 行情或策略计算失败
    if not isinstance(result, tuple):

        print(result)

        sys.exit(1)

    msg, signal, trade_date = result

    print(msg)

    # =========================
    # 交易日期去重
    # =========================

    last_trade_date = get_last_trade_date()

    if last_trade_date and trade_date <= last_trade_date:

        print(
            f"⏭️ 跳过推送：交易日期 {trade_date} "
            f"已经推送过，或早于最近推送日期 {last_trade_date}。"
        )

        sys.exit(0)

    # =========================
    # 是否推送
    # =========================

    if not PUSH_NO_SIGNAL and signal == "⚪ 暂无交易信号":

        print("当前无交易信号，按照配置跳过推送。")

        sys.exit(0)

    # =========================
    # 推送成功后保存日期
    # =========================

    pushplus_send(msg)

    save_last_trade_date(trade_date)

    print("✅ 本次策略执行完成。")
