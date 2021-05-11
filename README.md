# Sales Forecasting of Product X
The ‘Sales Info’ contains the daily data of 2015-01-02-2017-08-27.  
Product X has a very long sales period, and its price is constant. Therefore, I guess it’s a classic product, like all white Air Force 1.  
We need to forecast the sales performance of product X in the upcoming months, which means we need to forecast $Sales Amount and # Sold Unit. Generally, they are highly correlated. So, our dependent variable is $Sales Amount or # Sold Unit.  
As we don’t have many other valid variables, I decided to use time series method. I conducted data preprocessing, performed the ADF and white noise tests. And I used ARIMA model to forecast the sales performance. 
