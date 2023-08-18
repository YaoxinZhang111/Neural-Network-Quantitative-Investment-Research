import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# 
# 




scaler=MinMaxScaler(feature_range=(0,1))
# 
# 
datas = []
pre_num = 10
HISTORY = 30
raw_data = pd.read_csv('hs300.csv')    
plist = raw_data.iloc[:,5:6].values
plist=list(scaler.fit_transform(plist))
for idx in range(1, len(plist) - HISTORY - pre_num):
    sample = []
    for i in range(HISTORY):
        sample.append(plist[idx + i - 1] )
    for n in range(pre_num):
        answer = plist[idx + HISTORY + n - 1] 
        sample.append(answer)
    datas.append(sample)
datas = pd.DataFrame(data=datas)
datas = datas.astype(np.float32)
datas.dropna(axis=0, how='any', inplace=True)
# 
# 
test_datas = []
raw_data = pd.read_csv('sz.000776.csv')    
plist = raw_data.iloc[:,5:6].values
plist=list(scaler.fit_transform(plist))
for idx in range(1, len(plist) - HISTORY - pre_num):
    sample = []
    for i in range(HISTORY):
        sample.append(plist[idx + i - 1] )
    for n in range(pre_num):
        answer = plist[idx + HISTORY + n - 1] 
    sample.append(answer)
    test_datas.append(sample)
test_datas = pd.DataFrame(data=test_datas)
test_datas = test_datas.astype(np.float32)
# 
# 
datas = shuffle(datas)
# 
# 
X_train ,y_train = datas.iloc[:20000,:HISTORY],datas.iloc[:20000,HISTORY:]
X_valid ,y_valid = datas.iloc[40000:42000,:HISTORY],datas.iloc[40000:42000,HISTORY:]
# 
X_test , y_test = test_datas.iloc[:,:HISTORY],test_datas.iloc[:,HISTORY:]
# 
X_train = np.reshape(np.array(X_train), (X_train.shape[0], HISTORY, 1))
X_test = np.reshape(np.array(X_test) ,(X_test.shape[0], HISTORY, 1))
X_valid = np.reshape(np.array(X_valid) ,(X_valid.shape[0], HISTORY, 1))
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_valid = y_valid.to_numpy()

# 
# 
y_pred = X_valid[:, -1]
standard=np.mean(keras.losses.mean_squared_error(y_valid, y_pred))

# model = keras.models.Sequential([
#     keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
#     keras.layers.LSTM(20, return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(pre_num))
# ]) 
model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.SimpleRNN(10)
]) 

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, y_train, epochs=50,
                     validation_data=(X_valid, y_valid))
model.save('model.h5')

print(standard)

# 
y_proba = model.predict(X_test)
y_proba = scaler.inverse_transform(y_proba)
y_test=scaler.inverse_transform(y_test)[:,-1]
y_proba = y_proba[:,-1]

# 
plt.plot(y_test, color='red', label=' Stock Price')
plt.plot(y_proba, color='blue', label=' Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()