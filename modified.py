# -*- coding: utf-8 -*-
"""
@author: MXX
"""

from statsmodels.tsa.arima_model import ARIMA
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller as ADF
import matplotlib.pyplot as plt 
import pandas as pd
from functools import wraps
import time

#decorator 
def timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ('spend time: {:.3f}s'.format(t1 - t0))
        return result
    return function_timer

@timer
def test(x,y):
    s = x + y
    time.sleep(1.5)
    return s

test(1,1)
    

#read data
#Please change file path
raw_data = pd.read_excel('C:/Users/MXX/Desktop/Projects/Sales Forecasting of Product X/Question 2.xlsx')

#data cleaning
#check NAN
print(raw_data.isnull().sum())
#delete rows unit = 0 & amount<>0
daily_data = raw_data[~raw_data['# Sold Unit'].isin([0])].reset_index()

#inventory 
daily_data['# Inventory Unit'][0] = 249 - sum(daily_data.iloc[0:7]['# Sold Unit'])
n = len(daily_data)
for i in range(1,n):
    if daily_data['# Inventory Unit'][i] == 0:
                  daily_data['# Inventory Unit'][i] = daily_data['# Inventory Unit'][i-1] \
                                                    - daily_data['# Sold Unit'][i]

inventory_short = daily_data.loc[daily_data['# Inventory Unit'] < 20]

#Outliers                                             
daily_data['$ Sales Amount'].plot() 
plt.title('daily Sales Amount' )
plt.show()    

mean = daily_data['$ Sales Amount'].mean()
std = daily_data['$ Sales Amount'].std()

for i in range(1,n):
    if daily_data['$ Sales Amount'][i] > mean+3*std:
        daily_data['$ Sales Amount'][i] = daily_data['$ Sales Amount'][i-1]   
        
daily_data['$ Sales Amount'].plot() 
plt.title('daily Sales Amount after outlier operaton' )
plt.show()   

#Convert daily data to monthly data

daily_data['Calendar Date (YYYYMMDD)']  = pd.to_datetime(daily_data['Calendar Date (YYYYMMDD)'],format ='%Y%m%d')                               

time_start=time.time()

monthly_data = daily_data.groupby([daily_data['Calendar Date (YYYYMMDD)'].apply(lambda x:x.year),\
                                   daily_data['Calendar Date (YYYYMMDD)'].apply(lambda x:x.month)]).sum()

ratios = daily_data.groupby([daily_data['Calendar Date (YYYYMMDD)'].apply(lambda x:x.year),\
                                   daily_data['Calendar Date (YYYYMMDD)'].apply(lambda x:x.month)]).count()

time_end=time.time()
print('spend time: {:.3f}s'.format(time_end-time_start))
# 0.031s

#Testing for the improvement 
test_data = daily_data.copy()
test_data['Calendar Date (YYYYMMDD)']  = pd.to_datetime(test_data['Calendar Date (YYYYMMDD)'],format ='%Y%m%d') 
                           
time_start=time.time()

import datetime
test_data['Calendar Date (YYYYMMDD)'] = test_data['Calendar Date (YYYYMMDD)'].apply(lambda x: datetime.datetime.strftime(x,'%Y-%m'))

test_monthly_data = test_data.groupby(test_data['Calendar Date (YYYYMMDD)']).sum()

test_ratios = test_data.groupby(test_data['Calendar Date (YYYYMMDD)']).count()

time_end=time.time()
print('test spend time: {:.3f}s'.format(time_end-time_start))
# 0.024s


#number of days adjustment   
ratios = ratios['index']
ratios.index = pd.period_range('2015-01','2017-08',freq='M')
    
monthly_sale_data = monthly_data[['$ Sales Amount','# Sold Unit']]                                
monthly_sale_data.columns = ['Amount','Unit']
monthly_sale_data.index = pd.period_range('2015-01','2017-08',freq='M')
monthly_sale_data = monthly_sale_data*30
monthly_sale_data = monthly_sale_data.div(ratios,axis=0)

#created average price by sales amount / sales units
sale_a = monthly_sale_data['Amount']
sale_u = monthly_sale_data['Unit']
monly_avg_price = sale_a/sale_u
monly_avg_p=pd.DataFrame({"avg price":monly_avg_price})
monthly_sale_data = pd.concat([monthly_sale_data,monly_avg_p],axis = 1)

#create inventory related variables
#Please change file path
inv_data=pd.read_excel('C:/Users/MXX/Desktop/Marketplace Analyst/inventory.xlsx')
inv_data.index = pd.period_range('2015-01','2017-08',freq='M')
monthly_sale_data = pd.concat([monthly_sale_data,inv_data],axis = 1)

#data overview
corrdf = monthly_sale_data.corr()

sale_a.plot()
plt.title('monthly Sales Amount' )
plt.show()

sale_u.plot()
plt.title('monthly Sales unit' )
plt.show()
###############################################################################################################
#Methodologies
#ARIMA
#ensure it’s stationary
#ADF test
print('sale amount ADF test：', ADF(sale_a))
print('sale unit ADF test：', ADF(sale_u))
D_a = sale_a.diff().dropna()
D_a.columns = ['sale amount diff']
D_u = sale_u.diff().dropna()
D_u.columns = ['sale unit diff']
print('sale amount diff ADF test：', ADF(D_a))
print('sale unit diff ADF test：', ADF(D_u))

#White noise test 
print('sale amount diff White noise test', acorr_ljungbox(D_a, lags=1))
print('sale unit diff White noise test', acorr_ljungbox(D_u, lags=1))

#implied variables
#ADF
#the previous period’s inventory
last_month_end_inv = monthly_sale_data['last_month_end']
last_month_end_inv = last_month_end_inv.drop(last_month_end_inv.index[0])
print('last_month_end ADF test：', ADF(last_month_end_inv))
D_inv = last_month_end_inv.diff().dropna()
print('last_month_end diff ADF test：', ADF(D_inv))
#average price
print('average price ADF test：', ADF(monly_avg_price))
D_p = monly_avg_price.diff().dropna()
print('average price diff ADF test：', ADF(D_p))

#White noise test
print('last_month_end diff White noise test', acorr_ljungbox(D_inv, lags=1))
print('monly_avg_price diff White noise test', acorr_ljungbox(D_p, lags=1))

#average price ARIMA

plot_acf(D_p).show()
plot_pacf(D_p).show()

monly_avg_price = monly_avg_price.astype(float)  
pmax = int(len(D_p)/10) 
qmax = int(len(D_p)/10) 

bic_matrix = []
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
        try: 
            tmp.append(ARIMA(monly_avg_price, (p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix) 
bic_matrix=bic_matrix.iloc[:,:3]
p,q = bic_matrix.stack().idxmin() 
print('p,q=：%s、%s' %(p,q)) 

model = ARIMA(monly_avg_price, (p,1,q)).fit() 
model.summary2() 
model.forecast(3) 

#linear Regression
monly_avg_price_ols = monly_avg_price.drop(monly_avg_price.index[0])
variebles=pd.DataFrame({"avg price":monly_avg_price_ols, "inv":last_month_end_inv})
sale_u_l = sale_u.drop(sale_u.index[0])
LR_model = sm.OLS(sale_u_l, sm.add_constant(variebles)).fit()
print (LR_model.summary())
LR_model.params

#PCA
pca = PCA(n_components='mle')
pca.fit(variebles)
variebles_PCA = pca.transform(variebles)

PCA1 = sm.OLS(sale_u_l, sm.add_constant(variebles_PCA)).fit()
print (PCA1.summary())
PCA2 = sm.OLS(sale_u_l, variebles_PCA).fit()
print(PCA2.summary())
                                   