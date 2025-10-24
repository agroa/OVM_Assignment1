import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy


data = []
data_weekly = []

with open('DailyData - STOCK_US_XNAS_AAPL.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Read the first row (column names)
    for row in reader:
        data.insert(0, row)

with open('WeeklyData - STOCK_US_XNAS_AAPL.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Read the first row (column names)
    for row in reader:
        data_weekly.insert(0, row)

dates = [row[0] for row in data]
open = [float(row[1]) for row in data]
high = [float(row[2]) for row in data]
low = [float(row[3]) for row in data]
close = [float(row[4]) for row in data]
volume = [row[5] for row in data]


close_weekly = [float(row[4]) for row in data_weekly]


#------------------------------ 1(a.1) --------------------------------#
t = np.arange(1, 754, 1)
plt.plot(t, close, marker='.', linestyle='-', label='Close Price')
plt.title('Stock Close Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.grid(True)
plt.legend()
plt.show()

#------------------------------ 1(a.2) --------------------------------#


returns = []
logreturns = []
t = np.arange(1, 753, 1)


for i in range(len(close) - 1):
    returns.append((close[i + 1] - close[i]) / close[i])
    logreturns.append(np.log(close[i + 1] / close[i]))

plt.plot(t, returns, marker='.', linestyle='-', label='Close Returns')
plt.title('Stock Return Over Time')
plt.xlabel('Date')
plt.ylabel('Return')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(t, logreturns, marker='.', linestyle='-', label='Close LogReturns')
plt.title('Stock LogReturn Over Time')
plt.xlabel('Date')
plt.ylabel('LogReturn')
plt.grid(True)
plt.legend()
plt.show()

#------------------------------ 1(b) --------------------------------#

# plt.hist(logreturns, label='Close LogReturns')
# plt.title('LogrReturn Histogram')
# plt.ylabel('LogReturn')
# plt.legend()
# plt.show()

mean = np.mean(logreturns)
std = np.std(logreturns, ddof = 1)

# scipy.stats.probplot(logreturns, dist="norm", plot=plt)
# plt.title("QQ Plot of Data")
# plt.show()

returns_weekly = []
logreturns_weekly = []
t = np.arange(1, 157, 1)


for i in range(len(close_weekly) - 1):
    returns_weekly.append((close_weekly[i + 1] - close_weekly[i]) / close_weekly[i])
    logreturns_weekly.append(np.log(close_weekly[i + 1] / close_weekly[i]))


# plt.plot(t, returns_weekly, marker='.', linestyle='-', label='Weekly Close Returns')
# plt.title('Weekly Stock Return Over Time')
# plt.xlabel('Week')
# plt.ylabel('Return')
# plt.grid(True)
# plt.legend()
# plt.show()

# plt.plot(t, logreturns_weekly, marker='.', linestyle='-', label='Close LogReturns')
# plt.title('Weekly Stock LogReturn Over Time')
# plt.xlabel('Week')
# plt.ylabel('LogReturn')
# plt.grid(True)
# plt.legend()
# plt.show()

mean = np.mean(logreturns_weekly)
std = np.std(logreturns_weekly, ddof = 1)

# scipy.stats.probplot(logreturns_weekly, dist="norm", plot=plt)
# plt.title("QQ Plot of Weekly LogReturns")
# plt.show()



