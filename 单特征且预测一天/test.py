
import numpy as np

import datetime
import baostock as bs
import pandas as pd



from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model

pre_num = 1
HISTORY = 10
raw_data = pd.read_csv('hs300.csv')    
plist = raw_data.iloc[:,5:6].values
scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(plist)


start_day =((datetime.datetime.now())+datetime.timedelta(days= -30)).strftime("%Y-%m-%d")


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])
model = load_model('model.h5',custom_objects={'last_time_step_mse': last_time_step_mse}) 



#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取交易日信息 ####
rs = bs.query_trade_dates(start_date=start_day)
#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=['date','trade_day'])
result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y%m%d')
#### 结果集输出到csv文件 ####   
trade_date = result[~(result['trade_day'] == '0')]
yesterday = trade_date.iloc[-2,0]
today = trade_date.iloc[-1,0]
correct = 0
incorrect = 0


stock_50 = bs.query_sz50_stocks()
stock_300 = bs.query_hs300_stocks()
stock_50r = stock_50.get_data()
stock_300r = stock_300.get_data()
#循环下载数据

bucket = []
code_list = []
# for codes in stock_300r["code"]:
#     codes_list = list(codes)
#     if codes_list[3] == '3' and codes_list[4] == '0' and codes_list[5] == '0':
#         pass
#     else:
#         if codes_list[3] == '6' and codes_list[4] == '8' and codes_list[5] == '8': 
#             pass
#         else:
#             code_list.append(codes)
for codes in stock_50r["code"]:
    code_list.append(codes)

for code in code_list:
    print("Downloading :" + code)
    # code = 'sz.000625'
    rs = bs.query_history_k_data_plus(code, 
        "date,code,close",
         start_date=start_day,
         frequency="d", adjustflag="3")
    
    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result['date'] = pd.to_datetime(result['date']).dt.strftime('%Y%m%d')
    plist = result.iloc[:,2:].values
    tr_plist=list(scaler.transform(plist))
    datas = []
    anwsers = []
    for idx in range(1, len(plist) - HISTORY - pre_num):
        sample = []
        for i in range(HISTORY):
            sample.append(tr_plist[idx + i - 1] )
        answer = plist[idx + HISTORY - 1]
        anwsers.append(answer)
        datas.append(sample)
   
    
    sample = pd.DataFrame(datas)
    sample = sample.astype(np.float32)
    sample = np.reshape(np.array(sample), (sample.shape[0], HISTORY, 1))

    
    possibility = model.predict(sample)
    possibility = scaler.inverse_transform(possibility)[:,-1]
    bucket.append((code,result.iloc[-1,2],possibility[-1],(float(result.iloc[-1,2])/float(result.iloc[-2,2]))-1,(possibility[-1]/float(result.iloc[-2,2]))-1))
    for i in range(len(anwsers)-1):
        if (float(anwsers[i+1]))-float((anwsers[i]))*(possibility[i+1]-float(anwsers[i])) > 0 :
            correct += 1
        if (float(anwsers[i+1])-float(anwsers[i]))*(possibility[i+1]-float(anwsers[i])) < 0 :
            incorrect +=1
bucket = pd.DataFrame(data = bucket,columns = ['code','real','predic','real_pct','pre_pct'])
bucket = bucket.sort_values('real_pct',ascending=False)
bucket.to_csv('try.csv')
print(bucket)
accuracy = correct/(correct+incorrect)
print('accuracy: %f' % accuracy)
#### 登出系统 ####
bs.logout()

