
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 




scaler=MinMaxScaler(feature_range=(0,1))
# 
# 

pre_num = 1
HISTORY = 60
seed = 62
# -------------------------------------------------------🤣🤣🤣--------------------------------------------------------------
stocks = pd.read_csv("stocks.csv")
zz500 = pd.read_csv("zz500.csv")
daily_price = pd.concat([stocks,zz500])
# daily_price.dropna(axis=0, how='any', inplace=True)
# daily_price.iloc[:,3:9] = scaler.fit_transform(daily_price.iloc[:,3:9])
clf =  scaler.fit(daily_price.iloc[:,3:9])
import pickle 
with open('clf.pickle','wb') as f: 
    pickle.dump(clf,f)
open_datas = []
high_datas = []
low_datas = []
close_datas = []
volume_datas = []
turn_datas = []
answer_datas = []
print(daily_price.shape)
for stock in daily_price['code'].unique():
    # 日期对齐
    data = pd.DataFrame(daily_price['date'].unique(), columns=['date'])  # 获取回测区间内所有交易日
    df = daily_price.query(f"code=='{stock}'")[
        ['date', 'open', 'high', 'low', 'close', 'volume', 'turn']]
    
    open_list = list(df.loc[:,'open'])
    high_list = list(df.loc[:,'high'])
    low_list = list(df.loc[:,'low'])
    close_list = list(df.loc[:,'close'])
    volume_list = list(df.loc[:,'volume'])
    turn_list = list(df.loc[:,'turn'])
    for idx in range(1,len(df)-HISTORY-pre_num):
        if df.iloc[idx-1:idx+HISTORY-1,:].isnull().values.any() == True:
            continue
        else:
            open_sample = []
            high_sample = []
            low_sample = []
            close_sample = []
            volume_sample = []
            turn_sample = []
            answer_sample =[]
            for i in range(HISTORY):
                open_sample.append(open_list[idx + i - 1] )
                high_sample.append(high_list[idx + i - 1] )
                low_sample.append(low_list[idx + i - 1] )
                close_sample.append(close_list[idx + i - 1] )
                volume_sample.append(volume_list[idx + i - 1] )
                turn_sample.append(turn_list[idx + i - 1] )
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
    print(f"{stock} Done !")

print("All stock Done !")

datas = np.array([open_datas,high_datas,low_datas,close_datas,volume_datas])
datas = datas.astype(np.float32)
datas = datas.transpose((1,2,0))
print(datas.shape)

answers = np.array(answer_datas)
answers = answers.astype(np.float32)
print(answers.shape)
np.random.seed(seed)
np.random.shuffle(datas)
np.random.seed(seed)
np.random.shuffle(answers)
# ------------------------------------------------------------------
# __________________________________________________________😵😵😵😵______________________________________________________________
open_datas = []
high_datas = []
low_datas = []
close_datas = []
volume_datas = []
turn_datas = []
answer_datas = []

raw_data = pd.read_csv('sh.601238.csv')    
raw_data.iloc[:,2:8] = scaler.transform(raw_data.iloc[:,2:8])
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

test_datas = np.array([open_datas,high_datas,low_datas,close_datas,volume_datas])
test_datas = test_datas.astype(np.float32)
test_datas = test_datas.transpose((1,2,0))
test_answers = np.array(answer_datas)
test_answers = test_answers.astype(np.float32)
# 
# 

# 
# 
X_train ,y_train = datas[:-1000,:,:],answers[:-1000,:]
X_valid ,y_valid = datas[-1000:,:],answers[-1000:,:]
# 
X_test , y_test = test_datas,test_answers
# 



# 🥲🥲🥲
# 
y_pred = X_valid[:, -1,3]
standard=np.mean(keras.losses.mean_squared_error(y_valid[:,-1], y_pred))

model = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True, input_shape=[None, 5]),
    keras.layers.LSTM(80, return_sequences=True),
    keras.layers.SimpleRNN(pre_num)
]) 




def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, y_train, epochs=50,
                     validation_data=(X_valid,y_valid))
model.save('model.h5')

print(standard)



loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# 🌚🌚🌚🌚
y_proba = model.predict(X_test)
print(y_proba.shape)
print(y_test.shape)
y_proba = np.reshape(y_proba[:,-1],(y_proba.shape[0],1))
y_proba = np.pad(y_proba,((0,0),(3,2)),'constant',constant_values=0)
y_proba = scaler.inverse_transform(y_proba)
y_proba = y_proba[:,3]
y_test = np.reshape(y_test[:,-1],(y_test.shape[0],1))
y_test = np.reshape(y_test,(y_test.shape[0],1))
y_test = np.pad(y_test,((0,0),(3,2)),'constant',constant_values=0)
y_test=scaler.inverse_transform(y_test)
y_test = y_test[:,3]

correct = 0
incorrect = 0

for i in range(len(y_proba)-1):
    if (float(y_test[i+1]))-float((y_test[i]))*(y_proba[i+1]-float(y_test[i])) > 0 :
        correct += 1
    if (float(y_test[i+1])-float(y_test[i]))*(y_proba[i+1]-float(y_test[i])) < 0 :
        incorrect +=1

accuracy = correct/(correct+incorrect)
print('accuracy: %f' % accuracy)

# 😵‍💫😵‍💫
plt.plot(y_test, color='red', label='MaoTai Stock Price')
plt.plot(y_proba, color='blue', label='Predicted MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()