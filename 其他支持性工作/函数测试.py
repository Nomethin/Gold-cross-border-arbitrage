import openpyxl
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import random
import math
import talib
import requests
import json
import 函数调用 as functions

#import tushare as ts
#ts.set_token("8738cde8fb3d07838ccd12f113fa3f28f3aff934e7e28d396b0d95ae")
# 初始化Tushare接口
#pro = ts.pro_api()
# 获取沪金（Au9999）的每日成交量数据
#df = pro.fut_daily(ts_code='Au9999.SHF', start_date='2022-01-01', end_date='2022-12-31', fields='trade_date,vol')
#没有权限是什么鬼

print(functions.diaoyong()) #测试一下函数调用


workbook = openpyxl.load_workbook('数据表格.xlsx')
sheet = workbook['整理后数据']

"""
记得下面的一系列东西也要在函数调用和测试里面改
"""
#这里是交易参数，可能后面还会补充开盘价，交易量等
trade_day = [cell.value for cell in sheet['A']][1:]
AU = [cell.value for cell in sheet['B']][1:]
AU_main = [cell.value for cell in sheet['D']][1:]
NAU = [cell.value for cell in sheet['C']][1:]
NAU_main = [cell.value for cell in sheet['E']][1:]
RMB = [cell.value for cell in sheet['H']][1:]
time_judge = [cell.value for cell in sheet['G']][1:]

#下面是基本面分析的参数
CBOE_Gold_ETF_Vol_Index = [cell.value for cell in sheet['M']][1:]
"""
芝加哥期权交易所的黄金交易所交易基金波动率指数，该指数可以提供关于市场情绪和风险偏好的参考。
较高的波动性可能意味着投资者对市场未来走势存在较大的不确定性，较低的波动性则可能意味着市场相对稳定。
"""

Fed_Funds_Effective_Rate = [cell.value for cell in sheet['N']][1:]
"""
联邦基金有效利率是指美国联邦基金市场上短期利率的加权平均利率。
较低的利率倾向于鼓励借款和投资，因为融资成本更低，从而刺激经济增长和消费活动。相反，较高的利率可能降低投资和消费的吸引力。
较低的利率倾向于提升黄金的吸引力，因为黄金被视为一种避险资产和对冲通胀的工具。低利率环境下，投资者可能更倾向于购买黄金作为一种保值手段。
"""

Nominal_Broad_Dollar_Index = [cell.value for cell in sheet['O']][1:]
"""
名义广义美元指数,它衡量了美元相对于其他货币的相对价值变化。基准值通常设定为100。
当指数上升时，意味着美元对其他货币升值，表明美元相对较强。相反，当指数下降时，意味着美元对其他货币贬值，表明美元相对较弱
（较强的美元通常被视为一种避险资产，可能在市场不稳定或风险加大时受到投资者的青睐。相反，较弱的美元可能被视为对冲通胀风险或寻求更高收益的机会。）
名义广义美元指数与黄金市场之间存在一定的反向关系。一般来说，当美元走强时，黄金价格往往下跌，因为黄金在美元强势时相对变得更昂贵。
"""

US_Unemployment_rate = [cell.value for cell in sheet['P']][1:]
"""
美国的失业率，每月更新一次，不必多说，是经济状况的风向标。(低失业率（通常在4%至6%之间）)
在很高的时候，经济不景气，增加黄金的避险需求和价格。特别是疫情期间。
"""


"""
70以上的RSI值表示超买（市场可能过热），而30以下的RSI值表示超卖（市场可能过冷）,下面是AU的RSI。
但是用套利这种方法好像用不到预测指标？
"""
def calculate_rsi(prices, period=5):
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi

"""
波动性指数
"""
def calculate_volatility_index(closing_prices, period=5): #波动性指数,较高的波动性指数表示价格波动较大，风险较高(对于套利就不一定了)
    # 将列表转换为Pandas Series对象
    closing_prices_series = pd.Series(closing_prices)
    # 计算真实范围（True Range）
    tr = np.abs(closing_prices_series - closing_prices_series.shift())
    atr = tr.rolling(period).mean()
    volatility_index = atr.std()
    # 返回波动性指数的列表
    return atr.tolist()
# 提取收盘价列并计算波动性指数的列表
#volatility_index_list = calculate_volatility_index(AU)


"""
一共三条的布林线，也称BOLL
"""

# 假设AU为沪金价格的列表
#AU_prices = [...]  # 你的沪金价格列表
# 将价格列表转换为Pandas Series对象
#prices_series = pd.Series(AU_prices)
def calculate_bollinger_bands(prices, window=20, num_std=2): #然后把转换完的AU扔进去，这个是一共三条组成的布林线
    # 计算移动平均线
    rolling_mean = prices.rolling(window=window).mean()

    # 计算标准差
    rolling_std = prices.rolling(window=window).std()

    # 计算上轨道线和下轨道线
    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)

    return rolling_mean, upper_band, lower_band

#print(len(AU))

"""
这个是MACD，MACD是移动平均收敛/发散指标（Moving Average Convergence Divergence）的缩写。
它是一种常用的技术分析指标，用于衡量价格趋势的动力和变化的速度。
MACD通过比较短期移动平均线（EMA）和长期移动平均线（EMA）之间的差异来生成信号。
"""

"""
MACD线：它是通过短期指数移动平均线（EMA）减去长期指数移动平均线（EMA）得到的差值。MACD线反映了短期和长期均线之间的差异，它在图表上呈现为一条曲线。
信号线：它是对MACD线进行平滑处理得到的平均值，通常使用指数移动平均线（EMA）来计算。信号线是MACD指标的观察线，它在图表上呈现为另一条曲线，用于判断价格趋势的转折点和交易信号。
柱状图：柱状图是MACD线减去信号线的差值，用于表示MACD指标的强度和变化。柱状图可以为正值或负值，正值表示MACD线高于信号线。柱状图的变化和形态可以用于判断价格动能的变化和可能的买卖信号
"""

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    def calculate_ema(data, period):
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema = np.convolve(data, weights, mode='full')[:len(data)]
        ema[:period] = ema[period]
        return ema

    short_ema = calculate_ema(data, short_period)
    long_ema = calculate_ema(data, long_period)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

macd_line, signal_line, histogram = calculate_macd(AU)

#三个都是1982

"""
这个是KDJ，用于判断价格走势的强弱和超买超卖情况。KDJ指标是根据随机指标（Stochastic Oscillator）演变而来的。
KDJ指标的取值范围通常在0到100之间
当K线从下方穿越D线时，视为买入信号；当K线从上方穿越D线时，视为卖出信号。J线的数值越大，代表价格走势的强势程度越高；J线的数值越小，代表价格走势的弱势程度越高
"""

def calculate_kdj(prices, n=9, m=3):
    high_prices = np.array(prices)
    low_prices = np.array(prices)
    close_prices = np.array(prices)

    highest_high = np.maximum.accumulate(high_prices)
    lowest_low = np.minimum.accumulate(low_prices)

    rsv = np.zeros_like(prices, dtype=float) # 初始化RSV为0

    for i in range(len(prices)):
        if highest_high[i] != lowest_low[i]:
            rsv[i] = (close_prices[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i]) * 100

    k_values = []
    d_values = []
    j_values = []

    for i in range(len(prices)):
        if i < n:
            k_values.append(np.nan)
            d_values.append(np.nan)
            j_values.append(np.nan)
        else:
            rsv_slice = rsv[i - n + 1:i + 1]
            k_value = np.mean(rsv_slice)
            k_values.append(k_value)

            if i < n + m - 1:
                d_values.append(np.nan)
                j_values.append(np.nan)
            else:
                d_value = np.mean(k_values[i - m + 1:i + 1])
                j_value = 3 * k_value - 2 * d_value
                d_values.append(d_value)
                j_values.append(j_value)

    return k_values, d_values, j_values

"""
Ak, Ad, Aj = calculate_kdj(AU)
NAk, NAd, NAj = calculate_kdj(NAU)
A_info_num = np.full(len(Ak), np.nan)
NA_info_num = np.full(len(Ak), np.nan)
kd_dif = np.full(len(Ak), np.nan)
for i in range(len(Ak)):
    A_info_num[i] = 1 / 2 * (Aj[i - 2] + Aj[i - 1]) if i > 1 else 0
    NA_info_num[i] = 1 / 2 * (NAj[i - 2] + NAj[i - 1]) if i > 1 else 0
    kd_dif[i] = Ak[i] - Ad[i]

print(Aj)
"""


#都是1982项


"""
W&R（Williams %R）是一种常用的技术分析指标，也称为威廉指标。
W&R指标的取值范围是-100到0，当W&R指标超过-20时，意味着市场处于超买状态，可能出现价格的回调或反转；当W&R指标达到0时，意味着市场处于超卖状态，可能出现价格的反弹或反转。
W&R指标常用的周期设置为14，但也可以根据具体情况进行调整。通常，当W&R指标从超买区域回落至超卖区域时，可以视为买入信号；而当W&R指标从超卖区域反弹至超买区域时，可以视为卖出信号。
"""

def calculate_wr(prices, n=14):
    high_prices = np.array(prices)
    low_prices = np.array(prices)
    close_prices = np.array(prices)

    highest_high = np.maximum.accumulate(high_prices)
    lowest_low = np.minimum.accumulate(low_prices)

    wr_values = []

    for i in range(len(prices)):
        if i < n-1:
            wr_values.append(np.nan)
        else:
            hn = highest_high[i-n+1:i+1]
            ln = lowest_low[i-n+1:i+1]
            c = close_prices[i]
            wr = (np.max(hn) - c) / (np.max(hn) - np.min(ln)) * (-100)
            wr_values.append(wr)

    return wr_values

wr_values = calculate_wr(AU)

"""
BIAS（乖离率）,BIAS指标的取值为百分比形式，可以为正数或负数。
当BIAS值为正时，意味着市场价格高于移动平均线，可能存在超买情况；当BIAS值为负时，意味着市场价格低于移动平均线，可能存在超卖情况。
一般来说，当BIAS指标值超过一定阈值，如±6%、±8%等，可以视为超买或超卖信号，意味着价格可能发生反转或调整
"""
#感觉这个数值算出来有点小怪

def calculate_bias_list(prices, n, initial_value=0):
    bias_list = []

    def calculate_bias(prices, n):
        ma = np.mean(prices[-n:])  # 计算最近n个周期的移动平均值
        bias = (prices[-1] - ma) / ma * 100  # 计算乖离率
        return bias

    # 对前n个周期设定初始值
    for i in range(n):
        bias_list.append(initial_value)

    # 计算从第n+1个周期开始的乖离率
    for i in range(n, len(prices)):
        bias = calculate_bias(prices[:i + 1], n)
        bias_list.append(bias)

    return bias_list

n = 12  # 移动平均线的周期
bias_list = calculate_bias_list(AU, n)



"""
DPO(去趋势价格震荡指标)
DPO指标的取值可以为正值或负值。正值表示当前价格高于参考基准，负值表示当前价格低于参考基准。
"""
def calculate_dpo(prices, period=20):
    dpo = np.zeros_like(prices)  # 创建与价格列表相同大小的零数组
    moving_average = np.mean(prices[:period])  # 计算移动平均线作为参考基准
    dpo[period:] = prices[period:] - moving_average  # 计算DPO值
    return dpo

dpo_values = calculate_dpo(AU)

#之后再试试TRIX

"""
指数移动平均值(EXPMA)。观察这些值的变化可以帮助我们判断价格的整体走势和趋势的变化。
如果指数移动平均值逐渐上升，表示价格整体上呈现上升趋势；如果指数移动平均值逐渐下降，表示价格整体上呈现下降趋势；
如果指数移动平均值在一个相对稳定的水平上波动，表示价格整体上呈现横盘震荡
"""

def calculate_expma(prices, period=5):
    expma = np.zeros_like(prices)  # 创建与价格列表等长的指数移动平均列表
    smooth_factor = 2 / (period + 1)  # 平滑因子

    # 计算初始值
    expma[0] = prices[0]

    # 计算后续的指数移动平均值
    for i in range(1, len(prices)):
        expma[i] = prices[i] * smooth_factor + expma[i-1] * (1 - smooth_factor)

    return expma

expma_values = calculate_expma(AU)

"""
def detect_price_breakout(prices, expma_values, threshold=2):
    # 计算价格的标准差
    price_std = np.std(prices)

    for i in range(len(prices)):
        if abs(prices[i] - expma_values[i]) > threshold * price_std:
            # 价格脱离振荡，执行相应的措施
            if prices[i] > expma_values[i]:
                print(f"价格上涨超过阈值，执行平仓操作")
                # 执行平仓操作
            else:
                print(f"价格下跌超过阈值，执行买入操作")
                # 执行买入操作
            # 其他操作...
"""
