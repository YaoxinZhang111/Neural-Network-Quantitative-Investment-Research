import numpy as np
import baostock as bs
import pickle
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)




code = 'sh.600519'
rs = bs.query_history_k_data_plus(code, 
    "date,code,open,high,low,close,volume,turn",
    start_date='2020-01-01', end_date='2022-6-30',
    frequency="d", adjustflag="3")
    
    #### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y%m%d')
    
#### 登出系统 ####
bs.logout()


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])
model = load_model('model.h5',custom_objects={'last_time_step_mse': last_time_step_mse}) 

with open('clf.pickle','rb') as f:  
    clf_load = pickle.load(f)  #加载模型
    f.close 

HISTORY = 10
pre_num = 1

open_datas = []
high_datas = []
low_datas = []
close_datas = []
volume_datas = []
turn_datas = []
answer_datas = []

raw_data = result    
raw_data.iloc[:,2:8] = clf_load.transform(raw_data.iloc[:,2:8])
print(raw_data)

open_list = list(raw_data.loc[:,'open'])
high_list = list(raw_data.loc[:,'high'])
low_list = list(raw_data.loc[:,'low'])
close_list = list(raw_data.loc[:,'close'])
volume_list = list(raw_data.loc[:,'volume'])
turn_list = list(raw_data.loc[:,'turn'])
for idx in range(1,len(raw_data)-HISTORY-pre_num):
    open_sample = []
    high_sample = []
    low_sample = []
    close_sample = []
    volume_sample = []
    turn_sample = []
    answer_sample = []
    for i in range(HISTORY):
        open_sample.append(open_list[idx + i - 1] )
        high_sample.append(high_list[idx + i - 1] )
        low_sample.append(low_list[idx + i - 1] )
        close_sample.append(close_list[idx + i - 1] )
        volume_sample.append(volume_list[idx + i - 1] )
        turn_sample.append(turn_list[idx + i -1])
    open_datas.append(open_sample)
    high_datas.append(high_sample)
    low_datas.append(low_sample)
    close_datas.append(close_sample)
    volume_datas.append(volume_sample)
    turn_datas.append(turn_sample)

    for n in range(pre_num):
        answer = close_list[idx + HISTORY + n - 1] 
        answer_sample.append(answer)
    answer_datas.append(answer_sample)

test_datas = np.array([open_datas,high_datas,low_datas,close_datas,volume_datas,turn_datas])
test_datas = test_datas.astype(np.float32)
test_datas = test_datas.transpose((1,2,0))
test_answers = np.array(answer_datas)
test_answers = test_answers.astype(np.float32)

X_test , y_test = test_datas,test_answers

y_proba = model.predict(X_test)
print(y_proba.shape)
print(y_test.shape)
y_proba = np.reshape(y_proba[:,-1],(y_proba.shape[0],1))
y_proba = np.pad(y_proba,((0,0),(3,2)),'constant',constant_values=0)
y_proba = clf_load.inverse_transform(y_proba)
y_proba = y_proba[:,3]
y_test = np.reshape(y_test[:,-1],(y_test.shape[0],1))
y_test = np.reshape(y_test,(y_test.shape[0],1))
y_test = np.pad(y_test,((0,0),(3,2)),'constant',constant_values=0)
y_test=clf_load.inverse_transform(y_test)
y_test = y_test[:,3]

correct = 0
incorrect = 0

for i in range(len(y_proba)-1):
    if (float(y_test[i+1]))-float((y_test[i]))*(y_proba[i+1]-float(y_proba[i])) > 0 :
        correct += 1
    if (float(y_test[i+1])-float(y_test[i]))*(y_proba[i+1]-float(y_proba[i])) < 0 :
        incorrect +=1

accuracy = correct/(correct+incorrect)
print('accuracy: %f' % accuracy)

plt.plot(y_test, color='red', label=' Stock Price')
plt.plot(y_proba, color='blue', label='Predicted  Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()