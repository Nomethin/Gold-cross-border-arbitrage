import openpyxl
import matplotlib.pyplot as plt
import statistics
import numpy

workbook = openpyxl.load_workbook('修改版回测.xlsx')
sheet = workbook['修改版']

AU = [cell.value for cell in sheet['B']][1:]

def Backtesting(AU): #如果前一天涨我就做多AU
    Trade_N=[] #n-1项
    for i in range(1,len(AU)):
        if AU[i]-AU[i-1]>0:
            P_ti=1  #P_ti表示在第i天交易的数量,可以随着策略调整而调整。
            Trade_N.append(P_ti)
        else:
            P_ti=-1
            Trade_N.append(P_ti)
    return Trade_N
AU_N=Backtesting(AU)
#print(AU_N)

def AU_Yield(AU,AU_N):#这个是沪金收益率，记住是n-2项
    AU_Yield=[]
    for i in range(1,len(AU)-1):
        temp_yield = ((AU[i] - AU[i - 1]) / AU[i - 1]) * (-1) * AU_N[i]
        AU_Yield.append(temp_yield)
    return AU_Yield
AU_Y=AU_Yield(AU,AU_N)
#print(AU_Y)

def Cumulative_Yield(AU_Y):
    Cumulative_Yield = [AU_Y[0]]

    for i in range(1, len(AU_Y)):
        cumulative_return = Cumulative_Yield [i - 1] + AU_Y[i]
        Cumulative_Yield.append(cumulative_return)
    return Cumulative_Yield
Cumulative_Y = Cumulative_Yield(AU_Y)
#print(Cumulative_Y)

growth_diff=[]
for i in range(len(AU_Y)):
    growth_diff.append(AU_Y[i]-0.00012)
day_fluc = statistics.stdev(growth_diff) #日波动率
aver_growth = statistics.mean(growth_diff)
Sharpe_Ratio = (aver_growth * ((250)**0.5))/day_fluc

#print(day_fluc)
#print(aver_growth)
#print(Sharpe_Ratio)

plt.style.use("dark_background")
plt.figure(num=None, figsize=(12,6), frameon=True)
plt.title("Cumulative Yield")
plt.plot(range(len(AU_Y)), Cumulative_Y, color='green', marker='o', linewidth=1, markersize=0.5)
#plt.show()