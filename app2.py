# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 17:06:03 2021

@author: vishnu

"""
import streamlit as st
from PIL import Image
st.title(" Prediction of Exchange Rates ")
image=Image.open("bg.jpg")
st.image(image, use_column_width=True)
import streamlit as st
import datetime as dt
 
def numOfDays(date1, date2):
    return (date2-date1).days
     
# Driver program
date1 = st.date_input("Enter Start date")
date2 = st.date_input("Enter End date")
n=numOfDays(date1, date2)
#n

import pandas as pd


# In[3]:


ex_rate = pd.read_excel("DEXINUS (1).xls")
#ex_rate.head()


# In[4]:


ex_rate.columns = ['Date','Dexinus']
#ex_rate.head()


# In[5]:


ex_rate = ex_rate.interpolate(method ='linear')
#ex_rate.head()


# In[6]:


#ex_rate.isnull().sum()


# In[7]:


import matplotlib.pyplot as plt
plt.plot(ex_rate.Dexinus)


# In[8]:


### LSTM are sensitive to the scale of the data. so we can apply MinMax scaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
rate = scaler.fit_transform(np.array(ex_rate.Dexinus).reshape(-1,1))
#rate


# In[9]:


#rate.shape


# In[10]:


### splitting the data into train and test
training_size = int(len(rate)*0.90)
test_size = len(rate)-training_size
train_data,test_data = rate[0:training_size,:],rate[training_size:len(rate),:1]


# In[11]:


#len(train_data), len(test_data)


# In[12]:


import numpy
# converting an array of values into a dataset matrix
def  create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a= dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX),np.array(dataY)


# In[13]:


time_step = 50
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)


# In[14]:


#print(X_train.shape), print(Y_train.shape)


# In[15]:


#print(X_test.shape), print(Y_test.shape)


# In[16]:


### Reshape input to be [Samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[17]:


### create the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[21]:


model = Sequential()
model.add(LSTM(150,return_sequences=True,input_shape=(50,1)))
model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[22]:


model.summary()


# In[23]:


model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs = 10,batch_size = 50,verbose=1)


# In[24]:


### Lets do the prediction and check the performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[25]:


### Trnasform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[27]:


### Calculate RMSE metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(scaler.inverse_transform(Y_train.reshape(1,-1)).reshape(-1,1),train_predict))


# In[28]:


math.sqrt(mean_squared_error(scaler.inverse_transform(Y_test.reshape(1,-1)).reshape(-1,1),test_predict))


# In[29]:


def MAPE(org, pred):
    temp = np.abs((org - pred)/org)*100
    return np.mean(temp)


# In[30]:


MAPE(scaler.inverse_transform(Y_test.reshape(1,-1)).reshape(-1,1),test_predict)


# In[32]:


### Plotting
# shift train predictions for plotting
look_back = 50
trainPredictPlot = np.empty_like(rate)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(rate)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(rate)-1, :] = test_predict
# Plot baseline and predictions
plt.figure(figsize=(12,5))
plt.plot(scaler.inverse_transform(rate))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[33]:


#len(test_data)


# In[34]:


x_input = test_data[1265-50:].reshape(1,-1)
#x_input.shape


# In[35]:


temp_input = list(x_input)
temp_input = temp_input[0].tolist()
#temp_input


# In[36]:


# Demonstrate prediction for next 30 days
from numpy import array
lst_output = []
n_steps = 50
i=0
while(i<30):
    if(len(temp_input)>50):
        x_input = np.array(temp_input[1:])
       # print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input,verbose=0)
       # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
  #      print(yhat[0])
        temp_input.extend(yhat[0].tolist())
   #     print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
#print(lst_output)


# In[67]:


day_new = np.arange(1,51)
day_pred = np.arange(51,81)


# In[68]:


import matplotlib.pyplot as plt


# In[69]:


#len(rate)


# In[70]:


rate1 = rate.tolist()
rate1.extend(lst_output)


# In[71]:


plt.plot(day_new,scaler.inverse_transform(rate[12599:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[72]:


#inv_trans = scaler.inverse_transform(lst_output)
#nv_trans


# In[73]:


plt.figure(figsize=(12,5))
rate1 = rate.tolist()
rate1.extend(lst_output)
plt.plot(rate1[100:])
plt.show()


future_dates = pd.date_range(start = date1, end = date2, freq = 'D')
future_df = pd.DataFrame()
future_df['Day'] = [i.day for i in future_dates]
future_df['Month'] = [i.month for i in future_dates]
future_df['Year'] = [i.year for i in future_dates]    
future_df['Series'] = np.arange(12650,(12650+len(future_dates)))
#future_df


# In[95]:
result = scaler.inverse_transform(lst_output)

results = pd.DataFrame(result)


# In[96]:


final_df=pd.concat([future_df,results],axis=1)


# In[97]:


final_df.pop('Series')


# In[98]:


if(st.button("Predict")):
    #result=pd.DataFrame({'months':future_dates,'DEXINUS':results})
    st.write(final_df)