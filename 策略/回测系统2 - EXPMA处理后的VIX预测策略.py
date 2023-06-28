import openpyxl
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import random
import math
import cntalib as talib
import 函数调用 as functions

"""
说明：基于失业率操作，我们添加了失业增长率，其能更好的反映美国的社会状态，加了VIX.
同时，我们对两个因素都进行了EXPMA处理（因为我感觉平均趋势能更好的反映问题）。
"""

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

CBOE_VIX_Index = [cell.value for cell in sheet['Q']][1:]
"""
芝加哥期权交易所波动率指数,通常被称为"恐慌指数",
反映了市场对未来30个自然日（约一个月）内的股票市场波动性的预期。VIX指数的值越高，表示市场预期的波动性越大，市场参与者对未来的不确定性和风险感到担忧。
VIX指数的值通常以百分比形式呈现,这里用来衡量战争因素。
"""



def Backtesting(AU, NAU, trade_day, time_judge, RMB, grid_interval):

    length = len(AU)
    ratio = np.full(length, np.nan)
    RSI_dif = np.full(length, np.nan)
    RSI_dif_logchange = np.full(length, np.nan)

    A_info_num = np.full(length, np.nan)
    NA_info_num = np.full(length, np.nan)
    Info_dif = np.full(length, np.nan)

    US_Unrate_EXPMA = functions.calculate_expma(US_Unemployment_rate)
    US_Unrate_EXPMA_ratio = np.full(length, np.nan) #美国失业率改变率

    CBOE_VIX_Index_EXPMA = functions.calculate_expma(CBOE_VIX_Index)
    VIX_EXPMA_change = np.full(length, np.nan)


    scant = 1  # 修改了
    vcant = 0

    p_change = np.full(length, np.nan)
    Trade_N = np.full(length, np.nan)
    position_list = np.zeros(length)

    positions = []
    # grid_interval = 0.0005
    stop_loss = 0.1

    AU_rsi_values = functions.__calculate_rsi(AU) #长度都和AU一致，为1982项
    NAU_rsi_values = functions.__calculate_rsi(NAU)

    Aj = functions.calculate_j(AU) #长度都和AU一致，为1982项
    NAj = functions.calculate_j(NAU)


    for i, d in enumerate(trade_day):  # i是明天，只能有i-1，i-2之类的,那么一共1982项
        # 制造ratio
        ratio[i] = (NAU[i] * RMB[i] / 31.1035) / AU[i]

        # 制造信号大小，这个要用i-2
        A_info_num[i] = 0.5 * math.log(Aj[i - 2] + Aj[i - 1]+0.0000001) if i > 1 else 0  #这个0.0000001是用来防止前n项的0造成的数学错误的
        NA_info_num[i] = 0.5 * math.log(NAj[i - 2] + NAj[i - 1]+0.0000001) if i > 1 else 0
        #制造info_dif
        Info_dif[i] = 1+(A_info_num[i]-NA_info_num[i])

        #制造美国失业率改变率
        US_Unrate_EXPMA_ratio [i] = (US_Unrate_EXPMA[i-1]/US_Unrate_EXPMA[i-31])-1 if i>30 else 0 #减去31是因为这个是月结的

        #制造VIX的倾向
        VIX_EXPMA_change[i] = CBOE_VIX_Index_EXPMA[i-1] - CBOE_VIX_Index_EXPMA [i-2] if i>1 else 0

        #制造RSI_dif
        RSI_dif[i] = NAU_rsi_values[i]- AU_rsi_values[i]

        #制造RSI_dif_logchange,使得差值的绝对值越大，交易数目就越大
        if i > 0:
            RSI_dif_log = math.log(abs(RSI_dif[i - 1]))
            RSI_dif_logchange[i] = RSI_dif_log * math.copysign(1, RSI_dif[i - 1])
        else:
            RSI_dif_logchange[i] = 0

        # 计算expma
        expma_values = functions.calculate_expma(ratio)

        # 制造p_change
        if i>0:
            if RSI_dif_logchange[i] / scant >= 1:
                p_change[i] = vcant
            elif RSI_dif_logchange[i] / scant <= -1:
                p_change[i] = -vcant
            else:
                p_change[i] = (RSI_dif_logchange[i] / scant)
        else:
            p_change[i] = 0

        """
        下面是进入交易的准备工作,记得要i-1开始，上面可以用i是因为这个不参与交易大小。或者说是用来建立list的
        """

        ratio_value = ratio[i-1] if i > 0 else 1
        prev_ratio_value = ratio[i - 2] if i > 1 else 1 #注意这两个，i-1是现在，i是明天的预测交易

        AU_price = AU[i]
        NAU_price = NAU[i]

        grid_levels = np.arange(ratio_value - grid_interval, ratio_value + grid_interval, grid_interval)
        pyramid_factor = 2 * (prev_ratio_value - np.min(grid_levels)) / (np.max(grid_levels) - np.min(grid_levels))

        if pyramid_factor > 2:
            pyramid_factor = 2

        change = p_change[i-1] * pyramid_factor * Info_dif[i-1] if i > 0 else 0 #注意是拿今天的推测明天
        if change > 2:
            change = 2
        elif change < -2:
            change = -2

        #这些制造用i是没啥问题的，毕竟上面没有用来参与交易，除了ratio_value这两个和change

        """
        下面是交易逻辑
        """

        #expma_trend代表了ratio的上升下降趋势
        if i > 1:
            expma_trend = expma_values[i - 1] - expma_values[i - 2]  # 计算出trend
        else:
            expma_trend = 0

        if (abs(US_Unrate_EXPMA[i-1])>7 or abs(VIX_EXPMA_change[i-1])>1) and US_Unemployment_rate[i-1]>5: #两部分，一是失业率，二是恐慌指数
            # 我们可以直接抄底，分清是左侧还是右侧
            if expma_trend>0: #判断为左侧
                position_list[i] = -1

            else: #判断为右侧
                position_list[i] = 1

        elif US_Unemployment_rate[i-1] > 9  or abs(VIX_EXPMA_change[i-1])>2: #太高了，清仓后就别动了，不要在这个时候开仓
            position_list[i] = 0 #清仓

        elif (US_Unemployment_rate[i-2] > 5 and US_Unemployment_rate[i-1] < 5) or abs(VIX_EXPMA_change[i-1])<0.01: #跌出界限，直接清仓
            position_list[i] = 0 #清仓

        else: #市场在振荡，我们使用网格
            if (prev_ratio_value < np.min(grid_levels)) or (
                    prev_ratio_value > np.max(grid_levels)):  # 这个if是满足网格交易的条件
                position = {'ratio': ratio_value}
                positions.append(position)

                """
                下面一串是EXPMA，假设有问题，快速应对
                """
                if expma_trend > 0.005:  # 通常trend 超过千分之五我们就可以说是有变化倾向（查的）
                    position_list[i] = -1  # 快速反仓：ratio涨了，即NAU相对变贵，此时应多做空NAU
                elif expma_trend < -0.005:
                    position_list[i] = 1  # 快速反仓
                else:  # 这个else后就是正常交易时间的流程了
                    if ratio_value > 1:  # change不一定大于零，之前忘记加条件了
                        if time_judge[i] == 1:
                            if ratio_value > 1 + 0.08738 / AU_price:  # i等于0时是不会交易的
                                if position_list[i - 1] + change < 1 and position_list[i - 1] >= -1 and position_list[
                                    i - 1] <= 1:  # 然后position_list[i - 1]也不一定是正的
                                    position_list[i] = min(max(position_list[i - 1] + change, -1),
                                                           1)  # position_list[i]是个预测值! position_list[i]是明天的，现在change已经用的是i-1了
                                elif position_list[i - 1] + change >= 1:  # 正向爆仓，此时change>0
                                    position_list[i] = 1
                                else:  # 反向爆仓
                                    position_list[i] = -1
                            else:
                                position_list[i] = position_list[i - 1]
                        else:
                            if ratio_value > 1 + 0.02597 / AU_price:
                                if position_list[i - 1] + change < 1 and position_list[i - 1] >= -1 and position_list[
                                    i - 1] <= 1:
                                    position_list[i] = min(max(position_list[i - 1] + change, -1),
                                                           1)  # 用来保证position_list只在-1到1之间
                                elif position_list[i - 1] + change >= 1:  # 正向爆仓
                                    position_list[i] = 1
                                else:  # 反向爆仓
                                    position_list[i] = -1
                            else:
                                position_list[i] = position_list[i - 1]
                    else:
                        if ratio_value < 1 - 0.001838 / AU_price - 0.49 * RMB[i] / (31.1035 * AU_price):
                            if position_list[i - 1] <= 1 and position_list[i - 1] - change > -1 and position_list[
                                i - 1] >= -1:
                                position_list[i] = min(max(position_list[i - 1] - change, -1), 1)
                            elif position_list[i - 1] - change > -1:  # 正向爆仓
                                position_list[i] = -1
                            else:  # 反向爆仓
                                position_list[i] = 1
                        else:
                            position_list[i] = position_list[i - 1]

                    if ratio_value < position['ratio'] * (1 - stop_loss):  # 这个是止损
                        position_list[i] -= change

                    """
                    下面这一堆if是通过RSI指数判定要不要平仓
                    """
                    if AU_rsi_values[i - 1] > 70 and NAU_rsi_values[i - 1] <= 70:  # 一方降，一方升或者不变，平仓
                        position_list[i] += change
                    elif AU_rsi_values[i - 1] < 30 and NAU_rsi_values[i - 1] >= 30:  # 一方升，一方降或者不变，平仓
                        position_list[i] -= change
                    elif (AU_rsi_values[i - 1] >= 30 and AU_rsi_values[i - 1] <= 70) and NAU_rsi_values[i - 1] < 30:
                        position_list[i] -= change
                    elif (AU_rsi_values[i - 1] >= 30 and AU_rsi_values[i - 1] <= 70) and NAU_rsi_values[i - 1] > 70:
                        position_list[i] += change

                    Trade_N[i] = position_list[i] - position_list[i - 1]  # 挪到这里来了，删掉了很多重复片段

            else:  # 不满足网格，不交易
                position_list[i] = position_list[i - 1]
                Trade_N[i] = 0

    data = {
        'trade_day': trade_day,
        'AU': AU,
        'expma_trend': expma_trend,
        "US_Unrate_ratio":US_Unrate_EXPMA,
        'ratio': ratio,
        'time_judge': time_judge,
        'p_change': p_change,
        'Trade_N': Trade_N,
        'position_list': position_list,
        "VIX_EXPMA_change":VIX_EXPMA_change
    }
    df = pd.DataFrame(data) #导出csv文件
    df.to_excel('output.xlsx', index=False)

    return Trade_N, position_list, ratio ,df

Small_Trade_N,Small_postion_list,ratio,df = Backtesting(AU, NAU, trade_day, time_judge, RMB, grid_interval = 0.0002)
Middle_Trade_N,Middle_postion_list,ratio,df = Backtesting(AU, NAU, trade_day, time_judge, RMB, grid_interval = 0.0021)
Big_Trade_N,Big_postion_list,ratio,df = Backtesting(AU, NAU, trade_day, time_judge, RMB, grid_interval = 0.0031)

Trade_N = []
for i in range(len(Small_Trade_N)):
    Trade_N.append((0.5*Small_Trade_N[i]+0.25*Middle_Trade_N[i]+0.25*Big_Trade_N[i]))

postion_list = []
for i in range(len(Small_postion_list)):
    postion_list.append((0.5*Small_postion_list[i]+0.25*Middle_postion_list[i]+0.25*Big_postion_list[i]))

"""
大网，中网，小网同时进行，把资金分为0.5,0.25,0.25。这个数值是怎么来的呢？使用ratio的max和min，以中心为对称轴一边分成三等分，
落在最里面的比例即为小网占比“0.5”（近似值），以此类推。此外，更好的方法是，可以删去几个过大过小的极端值再进行划分。
"""

#Trade_N,postion_list = Backtesting(AU,NAU,ratio,time_judge,RMB,p_change)
#print("交易向量是: {0}".format(Trade_N)) #长度均为n-1
#print("仓位情况是: {0}".format(postion_list[:-1]))
#print(len(AU))
#print(len(Trade_N))
#print(len(postion_list))
#print("交易和仓位长度是否一致（0为一致）: {0}".format(len(Trade_N)-len(postion_list))) #长度一致
#postion_list = postion_list[:-1]
#print(len(ratio)) #sum用来删不需要的分支
print(df) #只要这个df不报错，就没有未来函数

Nega_Trade = 0
Posi_Trade = 0
No_Trade = 0
for i in range(len(Trade_N)):
    if Trade_N[i]>0:
        Posi_Trade+=1
    elif Trade_N[i]<0:
        Nega_Trade+=1
    else:
        No_Trade+=1
print(Posi_Trade,Nega_Trade,No_Trade)



"""
上面的部分是交易的基本逻辑
"""

#[i-1]才是交易日？

def AU_Yield(AU,AU_main,NAU_main,postion_list):#这个是沪金收益率，记住是n-1项
    AU_Yield = []
    for i in range(1,len(AU)):
        if i==1:
            AU_Yield.append(0)
            continue
        if AU_main[i - 2] != AU_main[i-1] or NAU_main[i - 2] != NAU_main[i-1]:
            AU_Yield.append(0)
        else:
            temp_yield = round(((AU[i-1] / AU[i - 2]) - 1)*postion_list[i-2],9)
            AU_Yield.append(temp_yield)
    return AU_Yield
AU_Y = AU_Yield(AU,AU_main,NAU_main,postion_list)
#print("沪金收益率是: {0}".format(AU_Y))

def NAU_Yield(RMB,NAU,NAU_main,AU_main,postion_list):#这个是纽约金收益率，记住是n-1项
    NAU_Yield=[]
    for i in range(1,len(NAU)):
        if i==1:
            NAU_Yield.append(0)
            continue
        if NAU_main[i-2]!=NAU_main[i-1] or AU_main[i - 2] != AU_main[i-1]:
            NAU_Yield.append(0)
        else:
            temp_yield = round((((NAU[i-1]*RMB[i-1]) / (NAU[i - 2]*RMB[i-2])) - 1)*postion_list[i-2],9)
            NAU_Yield.append(temp_yield)
    return NAU_Yield
NAU_Y=NAU_Yield(RMB,NAU,NAU_main,AU_main,postion_list)
#print("纽约金收益率是: {0}".format(NAU_Y))

def final_Yield(AU_Y,NAU_Y):
    final_Y = []
    for i in range(len(AU_Y)):
        final_Y.append(NAU_Y[i]-AU_Y[i])
    return final_Y
final_Y = final_Yield(AU_Y,NAU_Y)

#print("综合收益率是: {0}".format(final_Y))
#print(len(final_Y))
#print(len(AU_Y))
#print(len(AU))

def Cumulative_Yield(final_Y):
    Cumulative_Yield = [final_Y[0]]
    for i in range(1,len(final_Y)):
        cumulative_return = Cumulative_Yield [i - 1] + final_Y[i]
        Cumulative_Yield.append(cumulative_return)
    return Cumulative_Yield
Cumulative_Y = Cumulative_Yield(final_Y)
#print("Cumulative_Yield is: {0}".format(Cumulative_Y))
#print(len(Cumulative_Y))


"""
上面的部分是各种收益率的计算
"""


max_dd,max_dd_duration = functions.max_drawdown(Cumulative_Y)
#print("max_drawdown is: {0}".format(max_dd))
#print("max_drawdown duration is: {0}".format(max_dd_duration))

def Sharpe_Ratio(final_Y):
    growth_diff = []
    for i in range(len(final_Y)):
        growth_diff.append(final_Y[i])
    day_fluc = statistics.stdev(growth_diff)  # 日波动率
    aver_growth = statistics.mean(growth_diff)
    Sharpe_Ratio = (aver_growth * 15.5) / day_fluc  # 夏普指数
    return Sharpe_Ratio
Sharpe_Ratio=Sharpe_Ratio(final_Y)

def Sortino_Ratio(final_Y):
    growth_diff = []
    for i in range(len(final_Y)):
        growth_diff.append(final_Y[i])
    downside_diff = []  # 用以计算下行标准差
    for i in range(len(final_Y)):
        if final_Y[i] < 0:
            downside_diff.append(final_Y[i])
    std_downside = statistics.stdev(downside_diff) #下行标准差
    aver_growth = statistics.mean(growth_diff)
    Sortino_Ratio = (aver_growth * 15.5) / std_downside  # 索提诺指数
    return Sortino_Ratio
Sortino_Ratio = Sortino_Ratio(final_Y)

def Calmar_Ratio(Cumulative_Y,AU_Y,max_dd):
    anner_return_rate = (Cumulative_Y[-1] / len(AU_Y)) * 365
    Calmar_Ratio = anner_return_rate / max_dd
    return Calmar_Ratio
Calmar_Ratio = Calmar_Ratio(Cumulative_Y,AU_Y,max_dd)

print("Sharpe_Ratio is: {0}".format(Sharpe_Ratio))
print("Sortino_Ratio is: {0}".format(Sortino_Ratio))
print("Calmar_Ratio is: {0}".format(Calmar_Ratio))

"""
上面的部分是各种评测数据的计算
"""

upper_list,lower_list = functions.Upper_Lower(AU,RMB,time_judge)
#print(upper_list)
#print(lower_list)

#这个是上下界的分布图，发现是在1.0004~1.00005，还有-0.9998~-0.9995
plt.figure(figsize=(13, 13))
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }
data1 = upper_list[1:]
data2 = lower_list[1:]
plt.plot(data1, label="upper list")
plt.plot(data2, label="lower list")
#plt.axhline(1, linestyle='--', color='green', lw=2) #插入水平线
# plt.axvline(10, linestyle='--', color='green', alpha=0.8) //插入垂直线
plt.ylabel('value', fontsize=20)
plt.xlabel("time",labelpad=8.5, fontsize=20)
plt.legend(fontsize=20)
#plt.show()



#这个是交易分布的散点图
plt.scatter(range(len(AU)), Trade_N, marker="o", c="red" , s=0.5)
plt.title("distribution of trade")
#plt.show()

#这个是累计收益率的图
plt.style.use("dark_background")
plt.figure(num=None, figsize=(12,6), frameon=True)
plt.title("Cumulative Yield")
plt.plot(range(len(AU_Y)), Cumulative_Y, color='green', marker='o', linewidth=1, markersize=0.5)
plt.show()

#这个是Ratio的图,可以用来检验网格交易是否正确
plt.style.use("dark_background")
plt.figure(num=None, figsize=(12,6), frameon=True)
plt.title("Ratio")
plt.plot(range(len(AU_Y)+1), ratio, color='green', marker='o', linewidth=1, markersize=0.5)
#plt.show()

"""
最后的这一部分是绘图
"""