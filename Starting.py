# -*- coding: utf-8 -*-
"""
Created on Tue May 14 00:39:47 2019

@author: Eduardo Castanho

This script handles the preprocessing of the data
"""

#Loading Libraries
import numpy as np
from sklearn.preprocessing import StandardScaler

#loading files
data_train =  np.loadtxt( 'training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )

data_test =  np.loadtxt('test.csv', delimiter=',', skiprows=1)

#placing nans in the correct places instead of -999.0
data_train = np.where(data_train==-999.0,np.nan, data_train) 
data_test = np.where(data_test==-999.0,np.nan, data_test) 

#normalizing
scaler = StandardScaler()
scaler.fit(data_train)
data_train_normalized = scaler.transform(data_train)
scaler = StandardScaler()
scaler.fit(data_test)
data_test_normalized = scaler.transform(data_test)

#placing 0's in place of nans.
data_train_normalized[np.isnan(data_train_normalized)]=0
data_test_normalized[np.isnan(data_test_normalized)]=0

#placing the ids, weights and labels correctly.
data_train_normalized[:,0] = data_train[:,0]
data_train_normalized[:,32] = data_train[:,32]
data_train_normalized[:,31] = data_train[:,31]
data_test_normalized[:,0]  = data_test[:,0]

#saving data in csv files.
np.savetxt("./data_train_normalized.csv",data_train_normalized,fmt='%s',delimiter=',')
np.savetxt("./data_test_normalized.csv",data_test_normalized,fmt='%s',delimiter=',')
