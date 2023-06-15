import openpyxl
import matplotlib.pyplot as plt
import statistics
import numpy
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

def AU_Backtesting(AU): #如果前一天涨我就做多AU
    Trade_N=[] #n-2项
    for i in range(2,len(AU)):
        if (AU[i-1]/AU[i-2])-1>0:
            P_ti=1  #P_ti表示在第i天交易的数量,可以随着策略调整而调整。
            Trade_N.append(P_ti)
        else:
            P_ti=-1
            Trade_N.append(P_ti)
    return Trade_N
#AU_N=AU_Backtesting(AU)
#print(AU_N)

def NAU_Backtesting(NAU): #5天动量，是n-6,i=6
    Trade_N=[]
    for i in range(6,len(NAU)):
        if (NAU[i]/NAU[i-5])-1>0:
            P_ti=1  #P_ti表示在第i天交易的数量,可以随着策略调整而调整。
            Trade_N.append(P_ti)
        else:
            P_ti=-1
            Trade_N.append(P_ti)
    return Trade_N
#NAU_N=NAU_Backtesting(NAU)

ratio = []
for i in range(len(AU)):
    ratio.append((NAU[i]*RMB[i]/31.1035)/AU[i])  #这里是(NAU*汇率/31.1035)/AU 在1左右浮动
print(ratio)


def Backtesting(AU,NAU,ratio,time_judge,RMB):
    Trade_N=[]
    postion_list=[0] #多一个0，记得结尾删一个
    position=0 #有仓位就要有收益，没仓位就没收益,咱不一定要满仓(为1，也就是100%)，1可以换成别的东西。
    for i in range(1,len(ratio)):
        if ratio[i]>1: #这边是做多AU的情况
            if time_judge[i]==1:  #判断是否在2020年4月9号前，即中国黄金交割费用变化前
                if ratio[i]> 1 + 0.08738/AU[i] and ratio[i-1]< 1 + 0.08738/AU[i-1]:
                    p_change = 1 #我把这个定为了0.5
                    if position < 1 and position >= -1: #这里的一半等于号很重要
                        position += p_change
                        postion_list.append(position)
                        P_ti = p_change
                    else:
                        postion_list.append(position)
                        P_ti = 0
                else:
                    postion_list.append(position)
                    P_ti = 0
            else:
                if ratio[i]> 1 + 0.02597/AU[i] and ratio[i-1]< 1 + 0.02597/AU[i-1]:
                    p_change = 1
                    if position < 1 and position >= -1:
                        position += p_change
                        postion_list.append(position)
                        P_ti = p_change
                    else:
                        postion_list.append(position)
                        P_ti = 0
                else:
                    postion_list.append(position)
                    P_ti = 0
        else: #这边是做多NAU的情况
            if ratio[i] < 1 - 0.001838 / AU[i] - 0.49*RMB[i]/(31.1035*AU[i]) and ratio[i - 1] > 1 - 0.001838 / AU[i-1] - 0.49*RMB[i-1]/(31.1035*AU[i-1]):
                p_change = 1
                if position <= 1 and position > -1:
                    position -= p_change
                    postion_list.append(position)
                    P_ti = -p_change
                else:
                    postion_list.append(position)
                    P_ti = 0
            else:
                postion_list.append(position)
                P_ti = 0
        Trade_N.append(P_ti)
    return Trade_N,postion_list

Trade_N,postion_list = Backtesting(AU,NAU,ratio,time_judge,RMB)
print(Trade_N) #长度均为n-1
print(postion_list[:-1])
print(len(AU))
print(len(Trade_N))
print(len(Trade_N)-len(postion_list[:-1])) #长度一致
postion_list = postion_list[:-1]


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
#print(AU_Y)

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
#print(NAU_Y)

def final_Yield(AU_Y,NAU_Y):
    final_Y = []
    for i in range(len(AU_Y)):
        final_Y.append(AU_Y[i]-NAU_Y[i])
    return final_Y
final_Y = final_Yield(AU_Y,NAU_Y)
print(final_Y)
#print(len(final_Y))
#print(len(AU_Y))
#print(len(AU))
#print(final_Y)

def Cumulative_Yield(final_Y):
    Cumulative_Yield = [final_Y[0]]
    for i in range(len(final_Y)):
        cumulative_return = Cumulative_Yield [i - 1] + final_Y[i]
        Cumulative_Yield.append(cumulative_return)
    return Cumulative_Yield
Cumulative_Y = Cumulative_Yield(final_Y)
#print(Cumulative_Y)

growth_diff=[]
for i in range(len(final_Y)):
    growth_diff.append(final_Y[i])
#print(growth_diff)
day_fluc = statistics.stdev(growth_diff) #日波动率
aver_growth = statistics.mean(growth_diff)
Sharpe_Ratio = (aver_growth * 15.5)/day_fluc

#print(day_fluc)
print(Sharpe_Ratio)

plt.style.use("dark_background")
plt.figure(num=None, figsize=(12,6), frameon=True)
plt.title("Cumulative Yield")
plt.plot(range(len(AU_Y)+1), Cumulative_Y, color='green', marker='o', linewidth=1, markersize=0.5)
#plt.show()