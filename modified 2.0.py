# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:11:54 2020

@author: MXX
"""

import pandas as pd
from functools import wraps
import time
import datetime


data = pd.read_csv('C:/Users/MXX/Desktop/Projects/Sales Forecasting of Product X/daily data.csv')

#expand data size
data_np = pd.np.array(data)
data_np = pd.np.tile(data_np, (1000, 1))

daily_data = pd.DataFrame(data_np)
daily_data.columns = list(data)


#decorator with input & function name 
#有input时相当于在普通装饰器外加一层（def & return)
def timer(unit):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = function(*args, **kwargs)
            t1 = time.time()
            print (f'{function.__name__} spend time: {t1 - t0:.3f} {unit} ')
            return result
        return wrapper
    return decorator


#Convert daily data to monthly data
@timer('s')    
def method1():
    daily_data1 = daily_data.copy() 
    daily_data1['Calendar Date (YYYYMMDD)']  = pd.to_datetime(daily_data1['Calendar Date (YYYYMMDD)'],format ='%Y%m%d')
    
    monthly_data = daily_data1.groupby([daily_data1['Calendar Date (YYYYMMDD)'].apply(lambda x:x.year),\
                                       daily_data1['Calendar Date (YYYYMMDD)'].apply(lambda x:x.month)]).sum()
    ratios = daily_data1.groupby([daily_data1['Calendar Date (YYYYMMDD)'].apply(lambda x:x.year),\
                                 daily_data1['Calendar Date (YYYYMMDD)'].apply(lambda x:x.month)]).count()
    
    return monthly_data,ratios

  
@timer('s') 
def method2():
    daily_data2 = daily_data.copy() 
    daily_data2['Calendar Date (YYYYMMDD)']  = pd.to_datetime(daily_data2['Calendar Date (YYYYMMDD)'],format ='%Y%m%d')
    daily_data2['Calendar Date (YYYYMMDD)'] = daily_data2['Calendar Date (YYYYMMDD)'].apply(lambda x: datetime.datetime.strftime(x,'%Y-%m'))

    monthly_data = daily_data2.groupby(['Calendar Date (YYYYMMDD)']).sum()
    ratios = daily_data2.groupby(['Calendar Date (YYYYMMDD)']).count()

    return monthly_data,ratios

@timer('s') 
def method3():
    daily_data3 = daily_data.copy() 
    month = (daily_data3['Calendar Date (YYYYMMDD)'] - daily_data3['Year']*10000)//100
    daily_data3.insert(2,'Month',month)
    
    monthly_data = daily_data3.groupby(['Year','Month']).sum()
    ratios = daily_data3.groupby(['Year','Month']).count()
    
    return monthly_data,ratios

@timer('s') 
def method4():
    daily_data4 = daily_data.copy() 
    daily_data4['Calendar Date (YYYYMMDD)']  = pd.to_datetime(daily_data4['Calendar Date (YYYYMMDD)'],format ='%Y%m%d')
    daily_data4.insert(2,'Month',daily_data4['Calendar Date (YYYYMMDD)'].dt.month) #
    
    monthly_data = daily_data4.groupby(['Year','Month']).sum()
    ratios = daily_data4.groupby(['Year','Month']).count()
    
    return monthly_data,ratios



m1, r1 = method1()  
#method1 spend time: 15.190s

m2, r2 = method2()
#spend time: 11.619s   

m3, r3 = method3()
#spend time: 5.788s

m4, r4 = method4()
#spend time: 5.583s