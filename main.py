import yfinance as yf
import pandas as pd
import requests
import os

# ================= 配置区 =================
PUSHPLUS_TOKEN = os.environ.get('PUSHPLUS_TOKEN')

def run_strategy():
    # 1. 拉取数据
    qqq = yf.download('QQQ', period='300d', progress=False)
    if len(qqq) < 250:
        return "数据拉取失败，请检查网络。"
    
    # 2. 计算均线
    qqq['SMA120'] = qqq['Close'].rolling(window=120).mean()
    qqq['SMA200'] = qqq['Close'].rolling(window=200).mean()
    
    # 3. 提取数据
    t = qqq.iloc[-1]
    t1 = qqq.iloc[-2]
    
    close_t, close_t1 = t['Close'].item(), t1['Close'].item()
    sma120_t, sma120_t1 = t['SMA120'].item(), t1['SMA120'].item()
    sma200_t, sma200_t1 = t['SMA200'].item(), t1['SMA200'].item()

    # 4. 核心条件判断
    is_golden_cross = sma120_t > sma200_t
    
    # 连续2日跌破200
    drop_200_2d = (close_t < sma200_t) and (close_t1 < sma200_t1)
    # 连续2日站上200
    above_200_2d = (close_t > sma200_t) and (close_t1 > sma200_t1)
    # 连续2日跌破120
    drop_120_2d = (close_t < sma120_t) and (close_t1 < sma120_t1)
    # 连续2日站上120
    above_120_2d = (close_t > sma120_t) and (close_t1 > sma120_t1)

    # 5. 匹配你的四大状态机
    if drop_200_2d:
        status = "状态一：大级别熊市避险"
        action = "清空所有杠杆仓位，100% 满仓切换至 SGOV。"
    
    elif above_200_2d and not is_golden_cross:
        status = "状态二：底部反弹/震荡期"
        action = "持有 QLD（2倍杠杆）。等待 SMA120 上穿 SMA200（金叉）。"
    
    elif not above_200_2d and not drop_200_2d:
        # 既没有连续2日跌破，也没有连续2日站上（在200线附近纠缠）
        status = "状态未知 / 纠缠区"
        action = "价格在200日线附近未形成连续2日确认，建议维持当前仓位，观望。"

    elif above_200_2d and is_golden_cross:
        # 已经满足站上200日线，且金叉
        if above_120_2d:
            status = "状态三：主升趋势期"
            action = "全仓切换至 TQQQ（3倍杠杆）。"
        else:
            # 金叉了，但没站稳120日线，或者跌破了120日线
            status = "状态四：趋势降级/防守（或底部反弹延续）"
            action = "持有 QLD（2倍杠杆）。若价格再次跌破200线，则切回 SGOV。"

    # 处理一个极其极端的补丁：如果直接从高处跌穿120和200线
    if drop_120_2d and drop_200_2d:
        status = "状态一：大级别熊市避险（急跌穿透）"
        action = "连续2日跌破120与200日线，直接清仓，切换至 SGOV。"

    # 6. 拼接播报消息
    msg = f"【QQQ 今日盘面】\n"
    msg += f"收盘价: {close_t:.2f}\n"
    msg += f"SMA120: {sma120_t:.2f}\n"
    msg += f"SMA200: {sma200_t:.2f}\n"
    msg += f"金叉状态: {'✅ 是' if is_golden_cross else '❌ 否'}\n\n"
    msg += f"【策略判定】\n"
    msg += f"📍 {status}\n"
    msg += f"👉 操作建议: {action}"
    
    return msg

def send_pushplus(title, content):
    if not PUSHPLUS_TOKEN:
        print("错误：未找到 PUSHPLUS_TOKEN。")
        return
    url = "http://www.pushplus.plus/send"
    data = {
        "token": PUSHPLUS_TOKEN,
        "title": title,
        "content": content,
        "template": "txt"
    }
    requests.post(url, json=data)

if __name__ == "__main__":
    result = run_strategy()
    print(result)
    send_pushplus("QQQ 策略每日播报", result)
