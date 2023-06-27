import openpyxl
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import random

workbook = openpyxl.load_workbook('对回测.xlsx')
sheet = workbook['1-最简单回测']

AU = [cell.value for cell in sheet['B']][1:]
AU_main = [cell.value for cell in sheet['J']][1:]
NAU = [cell.value for cell in sheet['K']][1:]
NAU_main = [cell.value for cell in sheet['L']][1:]
RMB = [cell.value for cell in sheet['G']][1:]
time_judge = [cell.value for cell in sheet['M']][1:]
#print(NAU_main)

"""
这上面是模块，excel基本数据
"""

ratio = []
for i in range(len(AU)):
    ratio.append((NAU[i]*RMB[i]/31.1035)/AU[i])  #这里是(NAU*汇率/31.1035)/AU 在1左右浮动
#print(ratio)

def Trade_return(ratio,time_judge,AU,NAU,RMB): #先不考虑击穿,不对称的return
    T_return=[]
    for i in range(len(AU)):
        if ratio[i] > 1:  # 这边是做多AU的情况
            if time_judge[i] == 1:  # 判断是否在2020年4月9号前，即中国黄金交割费用变化前
                T_return.append(((NAU[i] * RMB[i]) / 31.1) - AU[i] - 0.08738)
            else:
                T_return.append(((NAU[i] * RMB[i]) / 31.1) - AU[i] - 0.02597)
        else:
            T_return.append(AU[i] - ((NAU[i] * RMB[i]) / 31.1) - 0.001838 - (0.49 * RMB[i] / 31.1))
    return T_return
T_return  = Trade_return(ratio,time_judge,AU,NAU,RMB)
#print(T_return)

T_return_ratio=[] #这边把利润比做出来
for i in range(1,len(T_return)):
    T_return_ratio.append(T_return[i]/T_return[i-1])
#print(T_return_ratio)


def P_change_func(AU,NAU,ratio,time_judge,RMB,T_return_ratio): #先做一个最简单的，p_change=利润（不对称的）的前后比值的某些比率,n-1项，感觉比值作为change还是有点大
    p_change = []
    for i in range(len(T_return_ratio)):
        if T_return_ratio[i]/2>1:
            p_change.append(0.5)        #这个1可以进行修改，譬如说设成一个特定数，这样后面可以放条件
            continue
        elif T_return_ratio[i]/2<-1:
            p_change.append(-0.5)
            continue
        p_change.append(T_return_ratio[i]/2) #用这个防止超正负1
    return p_change
p_change = P_change_func(AU,NAU,ratio,time_judge,RMB,T_return_ratio)
#print(p_change)

"""
上面的这部分是进行交易仓位调整的策略部分
"""


def Backtesting(AU,NAU,ratio,time_judge,RMB,p_change):
    Trade_N=[]
    postion_list=[0] #多一个0，记得结尾删一个
    position=0 #有仓位就要有收益，没仓位就没收益,咱不一定要满仓(为1，也就是100%)，1可以换成别的东西。
    for i in range(1,len(ratio)):
        if ratio[i]>1: #这边是做多AU的情况
            if time_judge[i]==1:  #判断是否在2020年4月9号前，即中国黄金交割费用变化前
                if ratio[i]> 1 + 0.08738/AU[i] and ratio[i-1]< 1 + 0.08738/AU[i-1]: #击穿条件，可以随时改
                    change = p_change[i] #我把这个定为了0.5
                    if position+change < 1 and position >= -1: #这里的一半等于号很重要
                        position += change
                        postion_list.append(position)
                        P_ti = change
                    else:
                        postion_list.append(position)
                        P_ti = 0
                else:
                    postion_list.append(position)
                    P_ti = 0
            else:
                if ratio[i]> 1 + 0.02597/AU[i] and ratio[i-1]< 1 + 0.02597/AU[i-1]:
                    change = p_change[i]
                    if position+change < 1 and position >= -1:  #加减change保证仓位不过正负1
                        position += change
                        postion_list.append(position)
                        P_ti = change
                    else:
                        postion_list.append(position)
                        P_ti = 0
                else:
                    postion_list.append(position)
                    P_ti = 0
        else: #这边是做多NAU的情况
            if ratio[i] < 1 - 0.001838 / AU[i] - 0.49*RMB[i]/(31.1035*AU[i]) and ratio[i - 1] > 1 - 0.001838 / AU[i-1] - 0.49*RMB[i-1]/(31.1035*AU[i-1]):
                change = p_change[i]
                if position <= 1 and position-change > -1:
                    position -= change
                    postion_list.append(position)
                    P_ti = -change
                else:
                    postion_list.append(position)
                    P_ti = 0
            else:
                postion_list.append(position)
                P_ti = 0
        Trade_N.append(P_ti)
    return Trade_N,postion_list

Trade_N,postion_list = Backtesting(AU,NAU,ratio,time_judge,RMB,p_change)
#print("交易向量是: {0}".format(Trade_N)) #长度均为n-1
#print("仓位情况是: {0}".format(postion_list[:-1]))
#print(len(AU))
#print(len(Trade_N))
#print("交易和仓位长度是否一致（0为一致）: {0}".format(len(Trade_N)-len(postion_list[:-1]))) #长度一致
postion_list = postion_list[:-1]

"""
上面的部分是交易的基本逻辑
"""


def AU_Yield(AU,AU_main,NAU_main,postion_list):#这个是沪金收益率，记住是n-2项
    AU_Yield = []
    for i in range(1,len(AU)):
        if AU_main[i - 1] != AU_main[i] or NAU_main[i - 1] != NAU_main[i]:
            AU_Yield.append(0)
        else:

            temp_yield = round(((AU[i] / AU[i - 1]) - 1)*postion_list[i-1],9)
            AU_Yield.append(temp_yield)
    return AU_Yield
AU_Y = AU_Yield(AU,AU_main,NAU_main,postion_list)
#print("沪金收益率是: {0}".format(AU_Y))

def NAU_Yield(RMB,NAU,NAU_main,AU_main,postion_list):#这个是纽约金收益率，记住是n-2项
    NAU_Yield=[]
    for i in range(1,len(NAU)):
        if NAU_main[i-1]!=NAU_main[i] or AU_main[i - 1] != AU_main[i]:
            NAU_Yield.append(0)
            continue
        else:
            temp_yield = round((((NAU[i]*RMB[i]) / (NAU[i - 1]*RMB[i-1])) - 1)*postion_list[i-1],9)
            NAU_Yield.append(temp_yield)
    return NAU_Yield
NAU_Y=NAU_Yield(RMB,NAU,NAU_main,AU_main,postion_list)
#print("纽约金收益率是: {0}".format(NAU_Y))

def final_Yield(AU_Y,NAU_Y):
    final_Y = []
    for i in range(len(AU_Y)):
        final_Y.append(AU_Y[i]-NAU_Y[i])
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
print("max_drawdown is: {0}".format(max_dd))
print("max_drawdown duration is: {0}".format(max_dd_duration))

growth_diff=[]
for i in range(len(final_Y)):
    growth_diff.append(final_Y[i])
#print(growth_diff)
day_fluc = statistics.stdev(growth_diff) #日波动率
aver_growth = statistics.mean(growth_diff)
Sharpe_Ratio = (aver_growth * 15.5)/day_fluc #夏普指数

anner_return_rate = (Cumulative_Y[-1]/len(AU_Y))*365
Calmar_Ratio = anner_return_rate/max_dd

#print(day_fluc)
print("Sharpe_Ratio is: {0}".format(Sharpe_Ratio))
#print(anner_return_rate)
print("Calmar_Ratio is: {0}".format(Calmar_Ratio))

"""
上面的部分是各种评测数据的计算
"""

#这个是交易分布的散点图
plt.scatter(range(len(AU_Y)), Trade_N, marker="o", c="red" , s=0.5)
plt.title("distribution of trade")
#plt.show()

#这个是累计收益率的图
plt.style.use("dark_background")
plt.figure(num=None, figsize=(12,6), frameon=True)
plt.title("Cumulative Yield")
plt.plot(range(len(AU_Y)), Cumulative_Y, color='green', marker='o', linewidth=1, markersize=0.5)
#plt.show()

"""
最后的这一部分是绘图
"""