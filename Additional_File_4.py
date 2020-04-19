# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:47:33 2019

@author: tchaganga
"""

# Please read the comments in the codes to make this program runs smoothly on your desktop.
# In my case I have created in folder name PythonQM9. Download all the supplementary files into this folder.
# Create the subfolder TFL_WVD in the PythomQM9 folder for saving the WVD of molecules computes using matlab.
# Create the subfolder TFL_STFT in the PythomQM9 folder for saving the Spectrogram of molecules computes using matlab.
# Create the subfolder TFL_CWT in the PythomQM9 folder for saving the scalogram of molecules computes using matlab.
# Run the matlab script to generate the TFL of the molecules. Make sure to specify in your Matlab script where the will be save by uncomment the right folder.
# Once you have all the ingredients, you can run your Python script for training and testing.

# Check and install the libraries that your python version might not have. for example Prettytable, tqdm, etc...
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
from numpy import genfromtxt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prettytable import PrettyTable # Install prettytable library

# This part reads the images.
train_image = []
with open('Additional_File_5.csv') as f: # make sure the file Additional_File_5 (Additional Files) is within the same directory as your python code.
    reader = csv.reader(f)
    next(reader) # skip header
    data = []
    for row in tqdm(reader):
        data.append(row)
        data1 = data[-1]
        data2 = data1[0]
        img0 = image.load_img('Spectrogram_QM9_WVD_EI/'+data2, target_size=(64,48,3), grayscale=False) # replace this folder "Spectrogram_QM9_WVD_EI" with the forlder that contains the images. 
        img0 = image.img_to_array(img0)
        img = img0/255
        train_image.append(img)
X = np.array(train_image)

files = r'D:\PythonQM9\Additional_File_1.csv' # Read the 19 properties. Thus make sure to specify the file and path where it is stored.
datax = genfromtxt(files, dtype=float, delimiter=',', names=None)

# data preprocessing. Normalization by dividing with the max
YT = datax;
YTT = datax;
for n in range(19):
    y = datax[0:datax.shape[0],datax.shape[1]-1-n]
    maxy = y.max()
    datax[0:datax.shape[0],datax.shape[1]-1-n] = y/maxy

# Data spliting into training and testing. I am using 90% for training and 10% for testing. You can specify otherwise.
X_train, X_test, yy_train, yy_test = train_test_split(X, datax, random_state=42, test_size=0.1)

Ix_train = X_train.shape[0]
Ix_test = X_test.shape[0]

Iy_train = yy_train.shape[0]
Iy_test = yy_test.shape[0]

# Training step for each property.
fig = plt.figure()
xt = PrettyTable()
xt.field_names = ["mean","std","maeT", "rmseT", "R2T", "maeV", "rmseV", "R2V","maeAll", "rmseAll", "R2All"]
for n in range(19):
    # Training and testing set for the properties
    y_train = yy_train[0:yy_train.shape[0],yy_train.shape[1]-1-n]
    y_test = yy_test[0:yy_test.shape[0],yy_test.shape[1]-1-n]
    y = datax[0:datax.shape[0],datax.shape[1]-1-n]
    maxyy = YTT[0:datax.shape[0],datax.shape[1]-1-n]
    maxxa = np.absolute(maxyy)
    maxx = maxxa.max()
    
    # Deep CNN construction
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),activation='relu',input_shape=(64,48,3)))
    
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(1, activation='linear'))
    
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mae'])
    
    # Training
    history = model.fit(X_train, y_train, epochs=250, validation_data=(X_test, y_test))
    
    # Prediction
    preds = model.predict(X_test)
    trains = model.predict(X_train)
    allData = model.predict(X)
    
    # Errors and Correlation Coefficients
    diff = preds.flatten() - y_test
    percentDiff = (diff / y_test) * 100
    absPercentDiff = np.abs(percentDiff)
    mean = np.mean(maxyy)
    std = np.std(maxyy)
    mae = mean_absolute_error(preds.flatten()*maxx, y_test*maxx) # Testing set
    mse = mean_squared_error(preds.flatten()*maxx, y_test*maxx) # Testing set
    r2 = r2_score(preds.flatten()*maxx, y_test*maxx) # Testing set
    maet = mean_absolute_error(trains.flatten()*maxx, y_train*maxx) # Training set
    mset = mean_squared_error(trains.flatten()*maxx, y_train*maxx) # Training set
    r2t = r2_score(trains.flatten()*maxx, y_train*maxx) # Training set
    maeall = mean_absolute_error(allData.flatten()*maxx, y*maxx) # Training and Testing (all dataset)
    mseall = mean_squared_error(allData.flatten()*maxx, y*maxx) # Training and Testing (all dataset)
    r2all = r2_score(allData.flatten()*maxx, y*maxx) # Training and Testing (all dataset)
    
    # Display output
    xt.add_row([format(mean,".5f"),format(std,".5f"),format(maet,".5f"), format(np.sqrt(mset),".5f"), format(r2t,".5f"), format(mae,".5f"), format(np.sqrt(mse),".5f"), format(r2,".5f"), format(maeall,".5f"), format(np.sqrt(mseall),".5f"), format(r2all,".5f")])
    print(xt)
    np.append(YT,allData,axis = 1)
    
    # Scatter plot measure vs prediction
    plt.scatter(y*maxx, allData*maxx, alpha=0.1, marker='.', c='b')
    plt.xlabel('Predictions')
    plt.ylabel('Measures')
    plt.show()
    
    # Evolution MAE
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    
    # Evolution LOSS
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
plt.show()
print(xt)
np.savetxt("Results.csv", YT, delimiter=",")
np.savetxt("yy_train.csv", yy_train, delimiter=",")
np.savetxt("yy_test.csv", yy_test, delimiter=",")