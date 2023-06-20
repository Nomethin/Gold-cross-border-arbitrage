import openpyxl
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import random
import math

workbook = openpyxl.load_workbook('对回测.xlsx')
sheet = workbook['1-最简单回测']

AU = [cell.value for cell in sheet['B']][1:]
AU_main = [cell.value for cell in sheet['J']][1:]
NAU = [cell.value for cell in sheet['K']][1:]
NAU_main = [cell.value for cell in sheet['L']][1:]
RMB = [cell.value for cell in sheet['G']][1:]
time_judge = [cell.value for cell in sheet['M']][1:]
trade_day = [cell.value for cell in sheet['A']][1:]
#print(NAU_main)

#print(len(AU))

def Backtesting(AU, NAU, trade_day, time_judge, RMB, grid_interval):

    length = len(AU)
    ratio = np.full(length, np.nan)
    T_return = np.full(length, np.nan)
    T_return_logchange = np.full(length, np.nan)

    scant = 3.5  # 我们击穿的数量有限，要让交易尽可能的多但也不能浪费其潜力，换言之，让仓满了不交易的情况尽量少
    vcant = 0.8

    p_change = np.full(length, np.nan)
    Trade_N = np.full(length, np.nan)
    position_list = np.zeros(length)

    positions = []
    # grid_interval = 0.0005
    stop_loss = 0.1

    for i, d in enumerate(trade_day):  # i是今天，只能有i，i-1之类的,那么一共1982项
        # 制造ratio
        ratio[i] = (NAU[i] * RMB[i] / 31.1035) / AU[i]
        # 制造T_return
        if ratio[i] > 1:
            if time_judge[i] == 1:
                T_return[i] = ((NAU[i] * RMB[i]) / 31.1) - AU[i] - 0.08738
            else:
                T_return[i] = ((NAU[i] * RMB[i]) / 31.1) - AU[i] - 0.02597
        else:
            T_return[i] = AU[i] - ((NAU[i] * RMB[i]) / 31.1) - 0.001838 - (0.49 * RMB[i] / 31.1)
        #制造T_return_logchange
        if i > 0:
            T_return_log = math.log(abs(T_return[i] - T_return[i - 1]))
            T_return_logchange[i] = T_return_log * math.copysign(1, T_return[i] - T_return[i - 1])
        else:
            T_return_logchange[i] = 1
        # 制造p_change
        if i>0:
            if T_return_logchange[i] * math.copysign(1, T_return[i] - T_return[i - 1]) / scant >= 1:
                p_change[i] = vcant
            elif T_return_logchange[i] * math.copysign(1, T_return[i] - T_return[i - 1]) / scant <= -1:
                p_change[i] = -vcant
            else:
                p_change[i] = (T_return_logchange[i] / scant)
        else:
            p_change[i] = 0

        ratio_value = ratio[i]
        prev_ratio_value = ratio[i - 1] if i > 0 else 1
        AU_price = AU[i]
        NAU_price = NAU[i]

        grid_levels = np.arange(ratio_value - grid_interval, ratio_value + grid_interval, grid_interval)
        pyramid_factor = 2 * (prev_ratio_value - np.min(grid_levels)) / (np.max(grid_levels) - np.min(grid_levels))

        if pyramid_factor > 2:
            pyramid_factor = 2

        change = p_change[i] * pyramid_factor

        if change > 2:
            change = 2
        elif change < -2:
            change = -2

        if (prev_ratio_value < np.min(grid_levels)) or (prev_ratio_value > np.max(grid_levels)):
            position = {'ratio': ratio_value}
            positions.append(position)

            if ratio_value > 1:
                if time_judge[i] == 1:
                    if ratio_value > 1 + 0.08738 / AU_price: #i等于0时是不会交易的
                        if position_list[i - 1] + change < 1 and position_list[i - 1] >= -1:
                            position_list[i] = position_list[i - 1] + change
                            Trade_N[i] = change
                        else:
                            position_list[i] = 0.5
                            Trade_N[i] = 0.5 - position_list[i - 1]
                    else:
                        position_list[i] = position_list[i - 1]
                        Trade_N[i] = 0
                else:
                    if ratio_value > 1 + 0.02597 / AU_price:
                        if position_list[i - 1] + change < 1 and position_list[i - 1] >= -1:
                            position_list[i] = position_list[i - 1] + change
                            Trade_N[i] = change
                        else:
                            position_list[i] = 0.5
                            Trade_N[i] = 0.5 - position_list[i - 1]
                    else:
                        position_list[i] = position_list[i - 1]
                        Trade_N[i] = 0
            else:
                if ratio_value < 1 - 0.001838 / AU_price - 0.49 * RMB[i] / (31.1035 * AU_price):
                    if position_list[i - 1] <= 1 and position_list[i - 1] - change > -1:
                        position_list[i] = position_list[i - 1] - change
                        Trade_N[i] = -change
                    else:
                        position_list[i] = -0.5
                        Trade_N[i] = -0.5 - position_list[i - 1]
                else:
                    position_list[i] = position_list[i - 1]
                    Trade_N[i] = 0

            for position in positions:
                if ratio_value < position['ratio'] * (1 - stop_loss):
                    positions.remove(position)
                    position_list[i] -= change
                    Trade_N[i - 1] = change
        else:
            position_list[i] = position_list[i - 1]
            Trade_N[i] = 0

    data = {
        'trade_day': trade_day,
        'AU': AU,
        'RMB': RMB,
        'ratio': ratio,
        'time_judge': time_judge,
        'T_return': T_return,
        'T_return_logchange': T_return_logchange,
        'p_change': p_change,
        'Trade_N': Trade_N,
        'position_list': position_list
    }
    df = pd.DataFrame(data)

    return Trade_N, position_list, ratio ,df

Small_Trade_N,Small_postion_list,ratio,df = Backtesting(AU, NAU, trade_day, time_judge, RMB, grid_interval = 0.0005)
Middle_Trade_N,Middle_postion_list,ratio,df = Backtesting(AU, NAU, trade_day, time_judge, RMB, grid_interval = 0.002)
Big_Trade_N,Big_postion_list,ratio,df = Backtesting(AU, NAU, trade_day, time_judge, RMB, grid_interval = 0.0035)

Trade_N = []
for i in range(len(Small_Trade_N)):
    Trade_N.append((1/3)*(Small_Trade_N[i]+Middle_Trade_N[i]+Big_Trade_N[i]))

postion_list = []
for i in range(len(Small_postion_list)):
    postion_list.append((1/3)*(Small_postion_list[i]+Middle_postion_list[i]+Big_postion_list[i]))

"""
大网，中网，小网同时进行，把资金一分为三，当然可以不均匀分配
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

def max_drawdown(Cumulative_Y): #输出最大回撤和回撤持续天数
    max_dd = 0.0
    max_dd_duration = 0
    curr_dd_duration = 0
    peak = Cumulative_Y[0]

    for ret in Cumulative_Y:
        if ret > peak:
            peak = ret
            curr_dd_duration = 0
        else:
            curr_dd_duration += 1
        if peak != 0:
            drawdown = (peak - ret) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_duration = curr_dd_duration

    return max_dd, max_dd_duration
max_dd,max_dd_duration = max_drawdown(Cumulative_Y)
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
def Upper_Lower(AU,RMB,time_judge):
    upper_list = []
    lower_list = []
    for i in range(len(AU)):
        if time_judge[i] == 1:
            upper_list.append(1 + 0.08738 / AU[i])
            lower_list.append(1 - 0.001838 / AU[i] - 0.49 * RMB[i] / (31.1035 * AU[i]))
        else:
            upper_list.append(1 + 0.02597 / AU[i])
            lower_list.append(1 - 0.001838 / AU[i] - 0.49 * RMB[i] / (31.1035 * AU[i]))
    return upper_list,lower_list
upper_list,lower_list = Upper_Lower(AU,RMB,time_judge)
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