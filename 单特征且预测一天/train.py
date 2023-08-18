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
HISTORY = 10
raw_data = pd.read_csv('hs300.csv')    

plist = raw_data.iloc[:,5:6].values
plist=list(scaler.fit_transform(plist))
for idx in range(1, len(plist) - HISTORY - 1):
    sample = []
    for i in range(HISTORY):
        sample.append(plist[idx + i - 1] )
    answer = plist[idx + HISTORY - 1] 
    sample.append(answer)
    datas.append(sample)
datas = pd.DataFrame(data=datas)
datas = datas.astype(np.float32)
# 
# 
test_datas = []
raw_data = pd.read_csv('sz.000776.csv')    
plist = raw_data.iloc[:,5:6].values
plist=list(scaler.fit_transform(plist))
for idx in range(1, len(plist) - HISTORY - 1):
    sample = []
    for i in range(HISTORY):
        sample.append(plist[idx + i - 1] )
    answer = plist[idx + HISTORY - 1] 
    sample.append(answer)
    test_datas.append(sample)
test_datas = pd.DataFrame(data=test_datas)
test_datas = test_datas.astype(np.float32)
# 
# 
datas = shuffle(datas)
# 
# 
X_train ,y_train = datas.iloc[:2000,:-1],datas.iloc[:2000,-1]
X_valid ,y_valid = datas.iloc[20000:24000,:-1],datas.iloc[20000:24000,-1]
# 
X_test , y_test = test_datas.iloc[:,:-1],test_datas.iloc[:,-1]
# （样本数 * 时间步长 * 纬度）
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
print(standard)
# model = keras.models.Sequential([
#     keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
#     keras.layers.LSTM(20, return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(1))
# ]) 
model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.SimpleRNN(1)
]) 
# 
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, y_train, epochs=5,
                     validation_data=(X_valid, y_valid))
# model.save('model.h5')
loss=model.evaluate(X_test, y_test)
print(loss)
# 
y_proba = model.predict(X_test)
y_proba = scaler.inverse_transform(y_proba)
y_test=scaler.inverse_transform([y_test])[0]
# 
plt.plot(y_test, color='red', label='MaoTai Stock Price')
plt.plot(y_proba, color='blue', label='Predicted MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()