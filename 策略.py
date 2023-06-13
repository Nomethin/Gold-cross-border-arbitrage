import openpyxl
workbook = openpyxl.load_workbook('修改版回测.xlsx')
sheet = workbook['修改版']


AU = [cell.value for cell in sheet['B']][1:]
NAU = [cell.value for cell in sheet['C']][1:]
RMB = [cell.value for cell in sheet['D']][1:]
time_judge = [cell.value for cell in sheet['H']][1:] #这个我就不用其他数据处理了，直接拿来用
#上面是基本量（给的）
Contract_Change_AU = [cell.value for cell in sheet['R']][1:]
Contract_Change_NAU = [cell.value for cell in sheet['S']][1:]
#换不换主力合约
#print(Contract_Change_AU)

trader_ind = []
for i in range(len(AU)):
    trader_ind.append((NAU[i]*RMB[i])/31.1-AU[i])
#这里是方便计算用的（交易子）
#print(trader_ind)

def Trade_Direction(RMB): #这个函数写了交易方向
    Trade_Direction = []
    for i in range(len(RMB)):
        if time_judge[i]==1:
            if trader_ind[i]>0:
                if abs(trader_ind[i])>0.08738:
                    Trade_Direction.append(1)
                else:
                    Trade_Direction.append(0)
            else:
                if abs(trader_ind[i])>0.001838+(0.49*RMB[i]/31.1):
                    Trade_Direction.append(-1)
                else:
                    Trade_Direction.append(0)
        else:
            if trader_ind[i] > 0:
                if abs(trader_ind[i]) > 0.02597:
                    Trade_Direction.append(1)
                else:
                    Trade_Direction.append(0)
            else:
                if abs(trader_ind[i]) > 0.001838 + (0.49 * RMB[i] / 31.1):
                    Trade_Direction.append(-1)
                else:
                    Trade_Direction.append(0)
    return Trade_Direction
Trade_D=Trade_Direction(RMB)
#这个函数是交易方向

def Trade_Profit(RMB):#这个函数是交易利润，基本上和交易方向的函数一样
    Trade_Profit = []
    for i in range(len(RMB)):
        if time_judge[i]==1:
            if trader_ind[i]>0:
                if abs(trader_ind[i])>0.08738:
                    Trade_Profit.append(abs(trader_ind[i])-0.08738)
                else:
                    Trade_Profit.append(0)
            else:
                if abs(trader_ind[i])>0.001838+(0.49*RMB[i]/31.1):
                    Trade_Profit.append(abs(trader_ind[i])-0.001838-(0.49*RMB[i]/31.1))
                else:
                    Trade_Profit.append(0)
        else:
            if trader_ind[i] > 0:
                if abs(trader_ind[i]) > 0.02597:
                    Trade_Profit.append(abs(trader_ind[i])-0.02597)
                else:
                    Trade_Profit.append(0)
            else:
                if abs(trader_ind[i]) > 0.001838 + (0.49 * RMB[i] / 31.1):
                    Trade_Profit.append(abs(trader_ind[i])-0.001838-(0.49*RMB[i]/31.1))
                else:
                    Trade_Profit.append(0)
    return Trade_Profit
Trade_P=Trade_Profit(RMB)
#这个函数是交易利润

def RMB_Ratio(RMB): #这个函数是汇率增长率，记得是n-1项
    RMB_Ratio=[]
    for i in range(len(RMB)-1):
        RMB_Ratio.append(RMB[i+1]/RMB[i])
    return RMB_Ratio
RMB_R=RMB_Ratio(RMB)
#这个函数是汇率增长率

def Profit_Ratio(Trade_P): #这个函数是利润增长率
    Profit_Ratio = []
    for i in range(1, len(Trade_P)):
        if Trade_P[i] == 0:
            Profit_Ratio.append(0)
        elif Trade_P[i - 1] == 0:
            j = i - 2
            while j >= 0 and Trade_P[j] == 0:
                j -= 1
            if j >= 0:
                Profit_Ratio.append(Trade_P[i] / Trade_P[j])
            else:
                Profit_Ratio.append(0)
        else:
            Profit_Ratio.append(Trade_P[i] / Trade_P[i - 1])

    return Profit_Ratio
Profit_R=Profit_Ratio(Trade_P)
#这个函数是利润增长率

def Num_of_Transactions(trader_ind,Profit_R,RMB_R,Trade_D,Contract_Change_AU,Contract_Change_NAU):#我先做不含自身交易增长率的,n项
    Num_of_Transactions=[trader_ind[0]]
    for i in range(len(Profit_R)):
        if Trade_D==0:
            Num_of_Transactions.append(0)
        else:
            if Profit_R[i]<0.9:
                Num_of_Transactions.append(0.5 * Trade_D[i] * RMB_R[i])
            elif Profit_R[i]>15:
                Num_of_Transactions.append(0.3 * Trade_D[i] * RMB_R[i])
            else:
                sum = 0
                if Contract_Change_AU == True:
                    sum += 4.4
                else:
                    sum -= 0.88
                if Contract_Change_NAU == True:
                    sum += 1
                else:
                    sum -= 0.1
                sum += Trade_D[i] * Profit_R[i] + 0.08 * RMB_R[i]
                Num_of_Transactions.append(sum * RMB_R[i])
    return Num_of_Transactions
Num_of_Trans = Num_of_Transactions(trader_ind,Profit_R,RMB_R,Trade_D,Contract_Change_AU,Contract_Change_NAU)
#这个函数是交易数目

def AU_Yield(Contract_Change_AU,AU,Num_of_Trans):#这个是沪金收益率，记住是n-2项
    AU_Yield=[]
    for i in range(2,len(AU)):
        if Contract_Change_AU == True:
            temp_yield = (AU[i - 1] - AU[i - 2]) / AU[i - 2] * (-1) * Num_of_Trans[i - 1]
            AU_Yield.append(temp_yield)
        else:
            temp_yield = (AU[i] - AU[i - 1]) / AU[i - 1] * (-1) * Num_of_Trans[i]
            AU_Yield.append(temp_yield)
    return AU_Yield
AU_Y=AU_Yield(Contract_Change_AU,AU,Num_of_Trans)
#这个函数是沪金收益率

def NAU_Yield(Contract_Change_NAU,NAU,Num_of_Trans):#这个是纽约金收益率，记住是n-2项
    NAU_Yield=[]
    for i in range(2,len(NAU)):
        if Contract_Change_NAU == True:
            temp_yield = (NAU[i - 1] - NAU[i - 2]) / NAU[i - 2] * Num_of_Trans[i - 1]
            NAU_Yield.append(temp_yield)
        else:
            temp_yield = (NAU[i] - NAU[i - 1]) / NAU[i - 1] * Num_of_Trans[i]
            NAU_Yield.append(temp_yield)
    return NAU_Yield
NAU_Y=NAU_Yield(Contract_Change_NAU,NAU,Num_of_Trans)
#这个函数是纽约金收益率，基本和上面一样

Comprehensive_return_rate = []
for i in range(len(AU_Y)):
    Comprehensive_return_rate.append(AU_Y[i]+NAU_Y[i])
#算了一下综合收益率

def Cumulative_Yield(Comprehensive_return_rate):
    Cumulative_Yield = [Comprehensive_return_rate[0]]

    for i in range(1, len(Comprehensive_return_rate)):
        cumulative_return = Cumulative_Yield [i - 1] + Comprehensive_return_rate[i]
        Cumulative_Yield.append(cumulative_return)
    return Cumulative_Yield
Cumulative_Y = Cumulative_Yield(Comprehensive_return_rate)
#这个是累计收益率