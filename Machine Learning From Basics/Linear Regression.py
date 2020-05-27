# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:40:58 2020

@author: Surya
Linear Regression won't work if the data doesn't have any relation between X and Y
best fit line: Y=MX+b
M=slope b=y intercept
M = (mean(x)mean(y)-mean(xy))/(mean(x))^2-(mean(x^2))

b=mean(y)-m(mean(x))
"""
import pandas as pd
import quandl,datetime   #Data store
import math
import numpy as np
#scaling your data between -1 to 1
from sklearn import preprocessing,svm
from sklearn.model_selection import cross_validate #cross_validation split up your data and suffle the data so we don't have a biased data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
#pickeling serialization of any python object
import pickle

quandl.ApiConfig.api_key = 'xJRxMMzV-h7ssv_mkb9h'
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)   #Filling NaN with -9999 as we cannot work with Nan's

forecast_out = int(math.ceil(0.01*len(df)))  #math.ceil round up to near number
print("Days:",forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out) #column will be shifted up with -forecast predicting 10 Adj. Close around 10 days in future


X = np.array(df.drop(['label'],1)) #features, which is everything except label
X = preprocessing.scale(X)  #Scale the data

X = X[:-forecast_out]
X_lately = X[-forecast_out:] #what we will be predicting

df.dropna(inplace=True)
y= np.array(df['label']) #labels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #0.2 which means 20% of data as testing data
#X_train and y_train we used to fit our classifier

#classifier using linear regression
#clf = LinearRegression(n_jobs=-1)
#clf.fit(X_train, y_train)
#saving the training data which can be retrained every month
#with open('linearregresseion.pickle','wb') as f:
#    pickle.dump(clf, f)
    
pickle_in = open('linearregresseion.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
#classifier using svm
#clf = svm.SVR()
 #clf.fit(X_train, y_train)
#accuracy = clf.score(X_test, y_test)

#print("Accuracy of the model:",accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan  #specify that entire data is full Nan data

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)        #date stamp
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]  #index is date 
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()