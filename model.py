#!/usr/bin/env python
# coding: utf-8

# In[93]:
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()


# # Read data into CSV and do some data cleaning
# ##### (1) Read in housing CSV data
# ##### (2) Map values of ocean_proximity to integers
# ##### (3) Fill in missing data with mean values from other data points
# ##### (4) Normalize data with z-scores
# 

# In[151]:
housing_data = pd.read_csv("./data/housing.csv")
housing_data['ocean_proximity']=number.fit_transform(housing_data['ocean_proximity'].astype('str'))
housing_data.fillna(housing_data.mean(), inplace=True)
house_value_mean = housing_data["median_house_value"].mean()
house_value_sd = housing_data["median_house_value"].std()

housing_data = (housing_data - housing_data.mean()) / housing_data.std()

print(housing_data.tail())


# In[116]:
# Features for training
features = housing_data.columns
features = features.drop("median_house_value")


# In[118]:
# Split into training and testing data
train_portion = 0.8
train_rows = int(housing_data.shape[0] * train_portion)
shuffled_data = housing_data.sample(frac=1)
train_data = shuffled_data[:train_rows]
test_data = shuffled_data[train_rows:]


# In[119]:
# Divide training data and labels
train_labels = train_data["median_house_value"]
train_data = train_data[features]

test_labels = test_data["median_house_value"]
test_data = test_data[features]


# In[120]:
print(train_labels.tail())
print(train_data.tail())


# In[121]:
# Split between training and validation set
X_train, X_val, Y_train, Y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=2)


# In[122]:
print(X_train.head())
print(X_train.shape)


# In[138]:
# Model composition. 9 features, use relu activation for positive linear and random initialization with Xavier. 2 hidden layers
model = Sequential()
model.add(Dense(30, input_dim=9, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')


# In[139]:
history = model.fit(X_train, Y_train, epochs=40, validation_data=(X_val, Y_val), verbose=2)


# In[140]:
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

predict = model.predict(test_data)


# In[152]:
predictions = predict*house_value_sd + house_value_mean
print("Predictions for test housing prices is: {}".format(predictions))


# In[145]:
print("Mean squared error between test_labels and predictions is: {}".format(mean_squared_error(test_labels, predict)))

