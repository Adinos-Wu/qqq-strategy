import yfinance as yf
import pandas as pd
import requests
import os
import json

# ================= 配置区 =================
PUSHPLUS_TOKEN = os.environ.get('PUSHPLUS_TOKEN')
STATE_FILE = 'state.json'

# ================= 获取数据与策略逻辑 =================
def run_strategy():
    # 1. 下载数据 (拉取过去300天)
    qqq = yf.download('QQQ', period='300d', progress=False)
    if len(qqq) < 250:
        return "数据拉取失败，请检查网络或yfinance。"
    
    # 2. 计算均线
    qqq['SMA120'] = qqq['Close'].rolling(window=120).mean()
    qqq['SMA200'] = qqq['Close'].rolling(window=200).mean()
    
    # 3. 提取最新两天的数据 (T日 和 T-1日)
    # iloc[-1] 是最新一天，iloc[-2] 是前一天
    t = qqq.iloc[-1]
    t1 = qqq.iloc[-2]
    
    close_t, close_t1 = t['Close'].item(), t1['Close'].item()
    sma120_t, sma120_t1 = t['SMA120'].item(), t1['SMA120'].item()
    sma200_t, sma200_t1 = t['SMA200'].item(), t1['SMA200'].item()

    # 4. 定义布尔条件
    drop_200_2d = (close_t < sma200_t) and (close_t1 < sma200_t1)
    above_200_2d = (close_t > sma200_t) and (close_t1 > sma200_t1)
    drop_120_2d = (close_t < sma120_t) and (close_t1 < sma120_t1)
    above_120_2d = (close_t > sma120_t) and (close_t1 > sma120_t1)
    is_golden_cross = sma120_t > sma200_t

    # 5. 读取当前状态
    current_state = "SGOV"
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            current_state = json.load(f).get("state", "SGOV")

    new_state = current_state
    action_msg = "维持现状，无需操作。"

    # 6. 状态机逻辑
    if current_state == "SGOV":
        if above_200_2d:
            new_state = "QLD"
            action_msg = "QQQ连续2日站上200日线，解除避险，买入QLD。"
            if is_golden_cross and above_120_2d:
                new_state = "TQQQ"
                action_msg = "QQQ站上200日线，且120日线金叉200日线，连续2日站稳120日线，直接买入TQQQ！"
                
    elif current_state == "QLD":
        if drop_200_2d:
            new_state = "SGOV"
            action_msg = "QQQ连续2日跌破200日线，清仓杠杆，切换至SGOV。"
        elif is_golden_cross and above_120_2d:
            new_state = "TQQQ"
            action_msg = "金叉确认且站稳120日线，从QLD升级至TQQQ。"
            
    elif current_state == "TQQQ":
        if drop_120_2d:
            if drop_200_2d:
                new_state = "SGOV"
                action_msg = "QQQ连续2日跌破120日和200日线，暴跌避险，清仓切至SGOV！"
            else:
                new_state = "QLD"
                action_msg = "QQQ连续2日跌破120日线，降级防守，切换至QLD。"
                
    # 7. 保存新状态
    with open(STATE_FILE, 'w') as f:
        json.dump({"state": new_state}, f)

    # 8. 生成推送消息
    msg = f"【当前状态】: {current_state}\n【策略判断】: {action_msg}\n【操作建议】: {new_state}"
    return msg

# ================= 推送与执行 =================
def send_pushplus(title, content):
    if not PUSHPLUS_TOKEN:
        print("错误：未找到 PUSHPLUS_TOKEN，请检查 GitHub Secrets 配置。")
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
    send_pushplus("QQQ 趋势策略提醒", result)
