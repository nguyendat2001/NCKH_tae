from tkinter import Y
import numpy as np
import pandas as pd
import os

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



# labels = ["pose_chao","pose_1","pose_2","pose_3","pose_4"]
# labels = ["pose_chao","pose_1","pose_2","pose_3","pose_4","pose_5",
#           "pose_6","pose_7","pose_8","pose_9","pose_10",
#           "pose_11","pose_12","pose_13","pose_14","pose_15",
#           "pose_16","pose_17","pose_18"]


dt_1 = pd.read_csv("pose_1.txt")
dt_2 = pd.read_csv("pose_2.txt")
# dt_3 = pd.read_csv("dt_3.txt")
# dt_4 = pd.read_csv("dt_4.txt")
# dt_5a = pd.read_csv("dt_5a.txt")
dt_5b = pd.read_csv("pose_5b.txt")
dt_6 = pd.read_csv("pose_6.txt")

X = []
y = []
no_of_timesteps = 10

     
dataset = dt_1.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append('pose_1')


dataset = dt_2.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append('pose_2')
    
# dataset = dt_3.iloc[:,1:].values
# n_sample = len(dataset)
# for i in range(no_of_timesteps, n_sample):
#     X.append(dataset[i-no_of_timesteps:i,:])
#     y.append(2)
    
# dataset = dt_4.iloc[:,1:].values
# n_sample = len(dataset)
# for i in range(no_of_timesteps, n_sample):
#     X.append(dataset[i-no_of_timesteps:i,:])
#     labels.append(4)

# dataset = dt_5a.iloc[:,1:].values
# n_sample = len(dataset)
# for i in range(no_of_timesteps, n_sample):
#     X.append(dataset[i-no_of_timesteps:i,:])
#     y.append(3)
    
dataset = dt_5b.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append('pose_5b')
    
dataset = dt_6.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append('pose_6')

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

from sklearn.preprocessing import LabelBinarizer
import pickle

encoder = LabelBinarizer()
y = encoder.fit_transform(y)
print(y)

file = open('pix.data', 'wb')
# dump information to that file
pickle.dump((X,y), file)
# close the file
file.close()

def load_data():
    file = open('pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels

X,y = load_data()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model  = Sequential()
model.add(LSTM(units = 1024, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 512, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 256, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 128))
model.add(Dropout(0.2))

model.add(Dense(4, activation='softmax', name='predictions'))

model.compile(optimizer="adam", metrics = ['accuracy'], loss = 'categorical_crossentropy')

model.fit(X_train, y_train, epochs=50, batch_size=50,validation_data=(X_test, y_test))
model.save("softmax_L1.h5")

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
print(X.shape, y.shape);
model.summary()


