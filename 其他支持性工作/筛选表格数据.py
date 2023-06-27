import openpyxl
import xlrd
from xlutils.copy import copy
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import numpy as np
import random
import math
import cntalib as talib
import 函数调用 as functions

# 打开原始 .xls 文件
workbook = xlrd.open_workbook('Nominal Broad U.S. Dollar Index.xls') #改这个东西的名字就可以了
sheet = workbook.sheet_by_name('FRED Graph') #这个也是

data = []
target_values = []

# 读取日期和值，并创建字典
for row in range(1, sheet.nrows):
    date = sheet.cell_value(row, 0)
    value = sheet.cell_value(row, 1)
    data.append({'date': date, 'value': value})

# 读取目标日期
target_dates = []
for row in range(1, sheet.nrows):
    target_date = sheet.cell_value(row, 4)
    target_dates.append(target_date)

# 根据目标日期筛选对应的值
for target_date in target_dates:
    for item in data:
        if item['date'] == target_date:
            target_values.append(item['value'])
            break
    else:
        target_values.append('')  # 如果没有找到对应的值，添加空白格

# 修改值为 42 的数据为前两个数据的平均值
for i in range(len(target_values)):
    if target_values[i] == 42:
        if i >= 2:
            average = (target_values[i-1] + target_values[i-2]) / 2
            target_values[i] = average

# 打开原始 .xls 文件的副本并准备写入数据
output_workbook = copy(workbook)
output_sheet = output_workbook.get_sheet(0)

# 写入筛选出的值到第 F 列
output_sheet.write(0, 5, 'Value')  # 写入标题
for row in range(1, sheet.nrows):
    output_sheet.write(row, 5, target_values[row - 1])

output_workbook.save('Modified_Nominal Broad U.S. Dollar Index.xls') #记得改一下文件名