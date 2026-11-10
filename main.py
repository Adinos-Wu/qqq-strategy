import yfinance as yf
import pandas as pd
import requests
import os

from datetime import datetime
from zoneinfo import ZoneInfo


# =========================
# 基础配置
# =========================

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN")

# 是否推送无交易信号的日报
PUSH_NO_SIGNAL = True

PUSH_TIMEOUT = 15


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

        qqq = qqq.dropna(subset=["Close"])

        # 美东时间
        now_et = datetime.now(ZoneInfo("America/New_York"))
        today_et = now_et.date()

        # 美股当天尚未收盘时，剔除当天未完成的日线
        if now_et.hour < 16:
            qqq = qqq[pd.Index(qqq.index.date) < today_et]

        # 计算均线
        qqq["SMA120"] = qqq["Close"].rolling(120).mean()
        qqq["SMA200"] = qqq["Close"].rolling(200).mean()

        qqq = qqq.dropna(subset=["SMA120", "SMA200"])

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

        # 均线金叉状态
        golden_cross = sma120 > sma200

        # =========================
        # 交易信号
        # =========================

        if drop_200_2d:

            signal = "🔴 避险信号"
            action = "卖出 QLD / TQQQ，切换至 SGOV"

        elif above_200_2d:

            if golden_cross and above_120_2d:

                signal = "🟢 进攻信号"
                action = "切换至 TQQQ"

            elif not golden_cross:

                signal = "🟡 反弹参与信号"
                action = "切换至 QLD"

            else:

                signal = "🟡 趋势确认等待"
                action = "维持观察，暂不调整"

        elif drop_120_2d:

            signal = "🟡 降级信号"
            action = "卖出 TQQQ，切换至 QLD"

        elif above_120_2d and golden_cross:

            signal = "🟢 重新进攻信号"
            action = "切换至 TQQQ"

        else:

            signal = "⚪ 暂无交易信号"
            action = "维持原有配置"

        # =========================
        # 行情信息
        # =========================

        change_120 = (close / sma120 - 1) * 100
        change_200 = (close / sma200 - 1) * 100

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
            "SMA120 > SMA200（金叉）"
            if golden_cross
            else "SMA120 < SMA200（死叉）"
        )

        # =========================
        # 推送内容
        # =========================

        msg = (
            f"📊 QQQ 四状态趋势策略日报\n"
            f"📅 交易日：{trade_date}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📈 行情数据\n"
            f"QQQ 收盘价：${close:.2f}\n"
            f"SMA120：${sma120:.2f}（{change_120:+.2f}%）\n"
            f"SMA200：${sma200:.2f}（{change_200:+.2f}%）\n"
            f"均线关系：{cross_status}\n"
            f"价格位置：{position_120}，{position_200}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 交易信号\n"
            f"{signal}\n"
            f"🔄 操作建议：{action}\n"
        )

        return msg, signal

    except Exception as e:
        return f"❌ 策略运行失败：{str(e)}"


# =========================
# PushPlus 推送
# =========================

def pushplus_send(content):

    if not PUSHPLUS_TOKEN:
        print("未设置 PUSHPLUS_TOKEN，跳过推送。")
        return

    url = "https://www.pushplus.plus/send"

    data = {
        "token": PUSHPLUS_TOKEN,
        "title": "QQQ 四状态趋势策略日报",
        "content": content,
        "template": "txt"
    }

    response = requests.post(
        url,
        json=data,
        timeout=PUSH_TIMEOUT
    )

    response.raise_for_status()

    print("PushPlus 推送完成：", response.text)


# =========================
# 主程序
# =========================

if __name__ == "__main__":

    result = run_strategy()

    if isinstance(result, tuple):

        msg, signal = result

        print(msg)

        if PUSH_NO_SIGNAL or signal != "⚪ 暂无交易信号":
            pushplus_send(msg)

    else:

        print(result)
