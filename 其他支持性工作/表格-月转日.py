import xlrd
from xlutils.copy import copy

# 打开原始 .xls 文件
workbook = xlrd.open_workbook('UNRATE.xls')
sheet = workbook.sheet_by_name('FRED Graph')

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

# 根据目标日期筛选对应的月份值
for target_date in target_dates:
    year, month, _, _, _, _ = xlrd.xldate_as_tuple(target_date, workbook.datemode)
    month_start_date = xlrd.xldate.xldate_from_date_tuple((year, month, 1), workbook.datemode)

    value = None
    for item in data:
        if item['date'] == month_start_date:
            value = item['value']
            break

    target_values.append(value)

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

output_workbook.save('Modified_UNRATE.xls')