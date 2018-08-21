import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

samples = 10000
L1 = 1
L2 = 1

def end_effector_pose(t1,t2):
	return [L1*np.cos(t1)+L2*np.cos(t1+t2),L1*np.sin(t1)+L2*np.sin(t1+t2)]

X_train = np.pi/2*np.random.random_sample((samples,2))
Y_train = np.zeros((samples,2)) 
for i in range(samples):
	Y_train[i][0] = L1*np.cos(X_train[i][0])+L2*np.cos(X_train[i][0]+X_train[i][1])
	Y_train[i][1] = L1*np.sin(X_train[i][0])+L2*np.sin(X_train[i][0]+X_train[i][1])

model = Sequential()
model.add(Dense(200,activation='relu',input_dim=2))
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(2))
print(model.summary())

# Store Weights
weights_file = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
try:
	os.chdir('weights')
except:
	os.mkdir('weights')
	os.chdir('weights')
os.mkdir(weights_file)
os.chdir(weights_file)
path = os.getcwd()
path = os.path.join(path,'weights_{epoch}.h5f')

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='mean_squared_error',optimizer=sgd)

checkpointer = ModelCheckpoint(filepath=path,period=50)
model.fit(X_train,Y_train,epochs=200,batch_size=100,callbacks=[checkpointer])

X_test = np.pi/2*np.random.random_sample((100,2))
Y_test = np.zeros((100,2))
for i in range(100):
	pose = end_effector_pose(X_test[i][0],X_test[i][1])
	Y_test[i][0]=pose[0]
	Y_test[i][1]=pose[1]

score = model.evaluate(X_test,Y_test)
print(score)

X_check = np.zeros((1,2))
X_check1 = np.zeros((1,2))
X_check1[0][1]=np.pi/2
Y_check = model.predict(X_check)
Y_check1 = model.predict(X_check1)
print(Y_check)
print(Y_check1)