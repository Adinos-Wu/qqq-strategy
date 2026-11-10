
import yfinance as yf
import pandas as pd
import requests
import os

# ================= 配置区 =================

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN")

# 是否推送无交易信号的日报
PUSH_NO_SIGNAL = True

# PushPlus请求超时时间
PUSH_TIMEOUT = 15


def run_strategy():
    try:
        # 1. 获取QQQ行情
        qqq = yf.download(
            "QQQ",
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if qqq.empty:
            return "❌ 行情获取失败：QQQ数据为空。"

        # 兼容新版yfinance多层列名
        if isinstance(qqq.columns, pd.MultiIndex):
            qqq.columns = qqq.columns.get_level_values(0)

        qqq = qqq.dropna(subset=["Close"])

        if len(qqq) < 202:
            return "❌ 行情数据不足，无法可靠计算SMA200。"

        # 2. 计算均线
        qqq["SMA120"] = qqq["Close"].rolling(120).mean()
        qqq["SMA200"] = qqq["Close"].rolling(200).mean()

        qqq = qqq.dropna(subset=["SMA120", "SMA200"])

        if len(qqq) < 2:
            return "❌ 有效均线数据不足。"

        # 3. 提取最近两个交易日
        t = qqq.iloc[-1]
        t1 = qqq.iloc[-2]

        close_t = float(t["Close"])
        close_t1 = float(t1["Close"])

        sma120_t = float(t["SMA120"])
        sma120_t1 = float(t1["SMA120"])

        sma200_t = float(t["SMA200"])
        sma200_t1 = float(t1["SMA200"])

        trade_date = qqq.index[-1].strftime("%Y-%m-%d")

        # 4. 判断连续两日条件
        drop_200_2d = (
            close_t < sma200_t
            and close_t1 < sma200_t1
        )

        above_200_2d = (
            close_t > sma200_t
            and close_t1 > sma200_t1
        )

        drop_120_2d = (
            close_t < sma120_t
            and close_t1 < sma120_t1
        )

        above_120_2d = (
            close_t > sma120_t
            and close_t1 > sma120_t1
        )

        # 5. 判断金叉
        golden_cross = sma120_t > sma200_t

        # 6. 生成交易信号
        signals = []

        if drop_200_2d:
            signals.append(
                "🚨【避险信号｜最高优先级】\n"
                "QQQ连续2个交易日收盘价低于SMA200。\n\n"
                "📌 操作建议：\n"
                "卖出QLD/TQQQ，切换至SGOV。\n"
                "🛡️ 退出杠杆仓位，进入防守。"
            )

        elif above_200_2d:

            if golden_cross and above_120_2d:
                signals.append(
                    "🚀【进攻信号｜趋势确认】\n"
                    "QQQ连续2日站上SMA120，且SMA120高于SMA200。\n\n"
                    "📌 操作建议：\n"
                    "若此前持有QLD，可切换至TQQQ（3倍杠杆）。\n"
                    "🔥 进入主升趋势进攻阶段。"
                )

            elif not golden_cross:
                signals.append(
                    "📈【反弹参与信号】\n"
                    "QQQ连续2日站上SMA200，但SMA120仍低于或等于SMA200。\n\n"
                    "📌 操作建议：\n"
                    "若此前持有SGOV，可切换至QLD（2倍杠杆）。\n"
                    "⚖️ 参与反弹，但暂不使用3倍杠杆。"
                )

            else:
                signals.append(
                    "⏳【趋势确认等待】\n"
                    "QQQ连续2日站上SMA200，且SMA120高于SMA200，"
                    "但尚未连续2日站上SMA120。\n\n"
                    "📌 操作建议：\n"
                    "暂不升级至TQQQ，等待价格确认。"
                )

        elif drop_120_2d:
            signals.append(
                "⚠️【降级信号｜趋势转弱】\n"
                "QQQ连续2个交易日收盘价低于SMA120，"
                "但尚未连续2日跌破SMA200。\n\n"
                "📌 操作建议：\n"
                "若此前持有TQQQ，卖出并切换至QLD（2倍杠杆）。\n"
                "🛡️ 降低杠杆，等待趋势修复。"
            )

        elif above_120_2d and golden_cross:
            signals.append(
                "🔄【重新进攻信号】\n"
                "QQQ连续2日站上SMA120，且SMA120高于SMA200。\n\n"
                "📌 操作建议：\n"
                "若此前持有QLD，可切换至TQQQ（3倍杠杆）。"
            )

        # 7. 生成每日投资简报

        # 均线趋势描述
        if golden_cross:
            trend = "🟢 多头排列"
        else:
            trend = "🔴 空头排列"

        # 价格与均线的距离
        distance_120 = (close_t / sma120_t - 1) * 100
        distance_200 = (close_t / sma200_t - 1) * 100

        # 价格位置
        if close_t > sma120_t and close_t > sma200_t:
            price_position = "🚀 处于双均线上方"
        elif close_t > sma200_t:
            price_position = "🟡 位于200日线上方、120日线下方"
        elif close_t < sma200_t:
            price_position = "🔴 位于200日线下方"
        else:
            price_position = "⚪ 均线附近震荡"

        # 8. 拼接播报消息
        msg = (
            "━━━━━━━━━━━━━━━━━━\n"
            "📊 QQQ 每日趋势跟踪简报\n"
            "━━━━━━━━━━━━━━━━━━\n\n"

            f"📅 交易日期：{trade_date}\n"
            f"💰 QQQ 收盘价：${close_t:.2f}\n\n"

            "【📈 均线监测】\n"
            f"🔵 SMA120：${sma120_t:.2f}（价格偏离 {distance_120:+.2f}%）\n"
            f"🟠 SMA200：${sma200_t:.2f}（价格偏离 {distance_200:+.2f}%）\n"
            f"📐 均线关系：{trend}\n"
            f"📍 当前位置：{price_position}\n\n"

            "━━━━━━━━━━━━━━━━━━\n"
            "🎯 策略信号\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
        )

        if signals:
            msg += "\n\n".join(signals)
        else:
            msg += (
                "⏳【今日无交易信号】\n\n"
                "当前行情尚未满足新的连续2日确认条件。\n\n"
                "📌 操作建议：\n"
                "保持原有策略，不因单日波动贸然调整仓位。"
            )

        msg += (
            "\n\n━━━━━━━━━━━━━━━━━━\n"
            "⚠️ 风险提示\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "1️⃣ 本简报基于QQQ收盘数据计算，信号需收盘后确认。\n"
            "2️⃣ 信号不代表账户实际持仓，请结合当前仓位执行。\n"
            "3️⃣ 杠杆ETF存在波动损耗，注意控制风险。\n\n"
            "🤖 QQQ Trend Following System"
        )

        return msg

    except Exception as e:
        return f"❌ 策略运行失败：{type(e).__name__}: {e}"


def send_pushplus(title, content):
    if not PUSHPLUS_TOKEN:
        print("⚠️ 未设置PUSHPLUS_TOKEN，跳过推送。")
        return

    url = "https://www.pushplus.plus/send"

    data = {
        "token": PUSHPLUS_TOKEN,
        "title": title,
        "content": content,
        "template": "txt"
    }

    try:
        response = requests.post(
            url,
            json=data,
            timeout=PUSH_TIMEOUT
        )
        response.raise_for_status()
        print("📨 PushPlus响应：", response.text)

    except requests.RequestException as e:
        print("❌ PushPlus推送失败：", repr(e))


if __name__ == "__main__":
    result = run_strategy()

    print(result)

    if PUSH_NO_SIGNAL or "今日无交易信号" not in result:
        send_pushplus("📊 QQQ每日趋势跟踪简报", result)
