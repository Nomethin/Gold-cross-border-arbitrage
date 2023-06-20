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
#print(NAU_main)

"""
这上面是模块，excel基本数据
"""

ratio = [] #这个ratio没有办法改，因为要用于判断是否可以交易
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

T_return_logchange=[] #这边把利润的差做出来
for i in range(1,len(T_return)):
    if i == 1:
        T_return_logchange.append(1)
        continue
    T_return_log = math.log(abs(T_return[i-1]-T_return[i-2])) #尽量不使用ratio了,用差的绝对值的log值
    T_return_logchange.append(T_return_log*math.copysign(1,T_return[i-1]-T_return[i-2]))
#print(len(T_return_logchange))
#print(T_return_logchange)

"""
math.copysign(1,T_return[i-1]-T_return[i-2])是用来表示T_return[i-1]-T_return[i-2]的符号的
"""

def P_change_func(T_return_logchange,T_return):
    p_change = []
    scant = 2.2 #我们击穿的数量有限，要让交易尽可能的多但也不能浪费其潜力，换言之，让仓满了不交易的情况尽量少
    vcant = 0.8
    for i in range(len(T_return_logchange)):
        if T_return_logchange[i]*math.copysign(1,T_return[i-1]-T_return[i-2])/scant >= 1:
            p_change.append(vcant)        #这个vcant可以进行修改
            continue
        elif T_return_logchange[i]*math.copysign(1,T_return[i-1]-T_return[i-2])/scant <= -1:
            p_change.append(-vcant)
            continue
        p_change.append((T_return_logchange[i]/scant)) #用这个防止超正负1
    return p_change
p_change = P_change_func(T_return_logchange,T_return)
#print(len(p_change)-len(ratio))
#把它单独拿出来，即使p_change可能会大于1，但还是会被筛掉


"""
上面的这部分是进行交易仓位调整的策略部分
"""


def Backtesting(AU, NAU, ratio, time_judge, RMB, p_change, grid_interval): #这边ratio没有办法改因为它涉及交易上下界
    Trade_N = [] # 记录交易大小（包含方向）的列表
    position_list = [0] # 记录仓位的列表
    positions = [] #用来止损的记录列表
    #grid_interval = 0.0005 #网格大小
    stop_loss = 0.1  # 止损比例

    for i in range(1, len(ratio)):

        ratio_value = ratio[i-1]  #更新好了
        prev_ratio_value = ratio[i - 2]
        if i == 1:
            prev_ratio_value = 1
        AU_price = AU[i-1]
        NAU_price = NAU[i-1] #为了对称而写的
        grid_levels = np.arange(ratio_value - grid_interval, ratio_value + grid_interval, grid_interval)

        # 计算prev_ratio_value在grid_levels中的相对位置，从而计算pyramid_factor
        pyramid_factor = 2*(prev_ratio_value - np.min(grid_levels)) / (np.max(grid_levels) - np.min(grid_levels)) #在正中间时pyramid_factor正好为1
        if pyramid_factor>2: #因为这个也是用比值算出来的，太大的舍去吧
            pyramid_factor = 2
        change = p_change[i - 1] * pyramid_factor
        if change>2: #也可以在p_change里面改，就是删去极端值，一样的
            change = 2
        elif change<-2:
            change = -2
        # 检查价格是否越过网格
        if (prev_ratio_value < np.min(grid_levels)) or (prev_ratio_value > np.max(grid_levels)): #简化了一下条件
            position = {'ratio': ratio_value}
            positions.append(position)
            if ratio_value > 1:  # 买入AU
                if time_judge == 1:
                    if ratio_value > 1 + 0.08738 / AU_price:  # 判断上界条件
                        if position_list[-1] + change < 1 and position_list[-1] >= -1:
                            # 开仓买入
                            position_list.append(position_list[-1] + change)
                            Trade_N.append(change)
                        else:
                            # 开仓买入但爆仓，半仓
                            position_list.append(0.5)
                            Trade_N.append(0.5 - position_list[-1]) #拿现在仓位减去以前的
                    else:
                        # 无交易
                        position_list.append(position_list[-1])
                        Trade_N.append(0)
                else:
                    if ratio_value > 1 + 0.02597 / AU_price:  # 判断上界条件
                        if position_list[-1] + change < 1 and position_list[-1] >= -1:
                            # 开仓买入
                            position_list.append(position_list[-1] + change)
                            Trade_N.append(change)
                        else:
                            # 开仓买入但爆仓，半仓
                            position_list.append(0.5)
                            Trade_N.append(0.5 - position_list[-1])
                    else:
                        # 无交易
                        position_list.append(position_list[-1])
                        Trade_N.append(0)
            else:  # 卖出AU
                if ratio_value < 1 - 0.001838 / AU_price - 0.49 * RMB[i] / (31.1035 * AU_price):  # 判断下界条件
                    if position_list[-1] <= 1 and position_list[-1] - change > -1:
                        # 开仓卖出
                        position_list.append(position_list[-1] - change)
                        Trade_N.append(-change)
                    else:
                        # 开仓卖出但爆仓，半仓
                        position_list.append(-0.5)
                        Trade_N.append(-0.5 - position_list[-1]) #
                else:
                    # 无交易
                    position_list.append(position_list[-1])
                    Trade_N.append(0)
                # 检查是否触及止损或止盈目标并执行平仓操作
                for position in positions:
                    if  ratio_value < position['ratio'] * (1 - stop_loss):
                        positions.remove(position)
                        # 执行平仓操作
                        position_list[-1] -= change  # 减少持仓数量
                        Trade_N[i-1] = change
        else:
            # 无交易
            position_list.append(position_list[-1])
            Trade_N.append(0)

    return Trade_N, position_list

Small_Trade_N,Small_postion_list = Backtesting(AU,NAU,ratio,time_judge,RMB,p_change, grid_interval = 0.0005)
Middle_Trade_N,Middle_postion_list = Backtesting(AU,NAU,ratio,time_judge,RMB,p_change, grid_interval = 0.002)
Big_Trade_N,Big_postion_list = Backtesting(AU,NAU,ratio,time_judge,RMB,p_change, grid_interval = 0.0035)

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
print("交易向量是: {0}".format(Trade_N)) #长度均为n-1
print("仓位情况是: {0}".format(postion_list[:-1]))
#print(len(AU))
#print(len(Trade_N))
print("交易和仓位长度是否一致（0为一致）: {0}".format(len(Trade_N)-len(postion_list[:-1]))) #长度一致
postion_list = postion_list[:-1]
#print(sum) #sum用来删不需要的分支


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
plt.scatter(range(len(AU_Y)), Trade_N, marker="o", c="red" , s=0.5)
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