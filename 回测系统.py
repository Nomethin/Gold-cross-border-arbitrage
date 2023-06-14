import openpyxl
import matplotlib.pyplot as plt
import statistics
import numpy

workbook = openpyxl.load_workbook('对回测.xlsx')
sheet = workbook['1-最简单回测']

AU = [cell.value for cell in sheet['B']][1:]
NAU = [cell.value for cell in sheet['K']][1:]
NAU_main = [cell.value for cell in sheet['L']][1:]
#print(AU)
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
AU_N=AU_Backtesting(AU)
#print(AU_N)

def NAU_Backtesting(NAU): #10天动量，是n-11,i=11
    Trade_N=[]
    for i in range(11,len(NAU)):
        if (NAU[i]/NAU[i-10])-1>0:
            P_ti=1  #P_ti表示在第i天交易的数量,可以随着策略调整而调整。
            Trade_N.append(P_ti)
        else:
            P_ti=-1
            Trade_N.append(P_ti)
    return Trade_N
NAU_N=NAU_Backtesting(NAU)
print(NAU_N)
print(len(NAU))
print(len(NAU_N))
#(0.4)

def AU_Yield(AU,AU_N):#这个是沪金收益率，记住是n-2项
    AU_Yield=[]
    for i in range(2,len(AU_N)+2):
        temp_yield = round(((AU[i]/AU[i-1])-1) * AU_N[i-2],9)
        AU_Yield.append(temp_yield)
    return AU_Yield
AU_Y=AU_Yield(AU,AU_N)
#print(AU_Y)

def NAU_Yield(NAU,NAU_N,NAU_main):#这个是纽约金收益率，记住是n-11项
    NAU_Yield=[]
    for i in range(11,len(NAU)):
        if NAU_main[i-1]!=NAU_main[i]:
            NAU_Yield.append(0)
            continue
        else:
            temp_yield = round(((NAU[i]/NAU[i-1])-1) * NAU_N[i-12],9)
            NAU_Yield.append(temp_yield)
    return NAU_Yield
NAU_Y=NAU_Yield(NAU,NAU_N,NAU_main)
print(NAU_Y)
print(len(NAU))
print(len(NAU_Y))


def Cumulative_Yield(AU_Y):
    Cumulative_Yield = [AU_Y[0]]

    for i in range(len(AU_Y)):
        cumulative_return = Cumulative_Yield [i - 1] + AU_Y[i]
        Cumulative_Yield.append(cumulative_return)
    return Cumulative_Yield
Cumulative_Y = Cumulative_Yield(AU_Y)
#print(Cumulative_Y)

growth_diff=[]
for i in range(len(NAU_Y)):
    growth_diff.append(NAU_Y[i])
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