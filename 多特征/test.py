
import numpy as np
import pickle
import datetime
import baostock as bs
import pandas as pd



from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import load_model

pre_num = 10
HISTORY = 60


with open('clf.pickle','rb') as f:  
    clf_load = pickle.load(f)  #加载模型
    f.close 

start_day =  ((datetime.datetime.now())+datetime.timedelta(days= -1000)).strftime("%Y-%m-%d")


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
for codes in stock_300r["code"]:
    codes_list = list(codes)
    if codes_list[3] == '3' and codes_list[4] == '0' and codes_list[5] == '0':
        pass
    else:
        if codes_list[3] == '6' and codes_list[4] == '8' and codes_list[5] == '8': 
            pass
        else:
            code_list.append(codes)
for codes in stock_50r["code"]:
    code_list.append(codes)



for code in code_list:
    print("Downloading :" + code)
    # code = 'sz.000776'
    rs = bs.query_history_k_data_plus(code, 
        "date,code,open,high,low,close,volume,turn",
         start_date=start_day,
         frequency="d", adjustflag="3")
    
    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
    try:
        result = pd.DataFrame(data_list, columns=rs.fields)
        
        result.iloc[:,2:] = clf_load.transform(result.iloc[:,2:])
        
        open_datas = []
        high_datas = []
        low_datas = []
        close_datas = []
        volume_datas = []
        answer_datas = []
        open_list = list(result.loc[:,'open'])
        high_list = list(result.loc[:,'high'])
        low_list = list(result.loc[:,'low'])
        close_list = list(result.loc[:,'close'])
        volume_list = list(result.loc[:,'volume'])
        for idx in range(1, len(result) - HISTORY - pre_num):
            open_sample = []
            high_sample = []
            low_sample = []
            close_sample = []
            volume_sample = []
            for i in range(HISTORY):
                open_sample.append(open_list[idx + i - 1] )
                high_sample.append(high_list[idx + i - 1] )
                low_sample.append(low_list[idx + i - 1] )
                close_sample.append(close_list[idx + i - 1] )
                volume_sample.append(volume_list[idx + i - 1] )
        

            open_datas.append(open_sample)
            high_datas.append(high_sample)
            low_datas.append(low_sample)
            close_datas.append(close_sample)
            volume_datas.append(volume_sample)

        
        sample = np.array([open_datas,high_datas,low_datas,close_datas,volume_datas])
        sample = sample.astype(np.float32)
        sample = sample.transpose((1,2,0))
        possibility = model.predict(sample)
        possibility = np.reshape(possibility[:,-1],(possibility.shape[0],1))
        y_proba = np.pad(possibility,((0,0),(3,2)),'constant',constant_values=0)
        y_proba = clf_load.inverse_transform(y_proba)
        possibility = y_proba[:,3]  
        result.iloc[:,2:] = clf_load.inverse_transform(result.iloc[:,2:])
        
        for i in range(len(result)-HISTORY-pre_num-1):
            n = i + HISTORY + pre_num - 1
            if (float(result.iloc[n+1,5]))-float((result.iloc[n,5]))*(possibility[i]-float(result.iloc[n,5])) > 0 :
                correct += 1
            if (float(result.iloc[n+1,5])-float(result.iloc[n,5]))*(possibility[i]-float(result.iloc[n,5])) < 0 :
                incorrect +=1
        bucket.append((code,result.iloc[-1,5],possibility[-1],(float(result.iloc[-1,5])/float(result.iloc[-1-pre_num,5]))-1,(possibility[-1]/float(result.iloc[-1-pre_num,5]))-1))
        print('success:' + code)
    except:
        pass
bucket = pd.DataFrame(data = bucket,columns = ['code','real','predic','real_pct','pre_pct'])
bucket = bucket.sort_values('real_pct',ascending=False)
bucket.to_csv('try.csv')
accuracy = correct/(correct+incorrect)
print(bucket)
print('accuracy: %f' % accuracy)
#### 登出系统 ####
bs.logout()


