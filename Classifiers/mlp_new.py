# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% Starting out
import random,string,math,csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from pylab import *

# MultiLayer Perceptron Classifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import metrics
#from tensorflow.keras.callbacks import EarlyStopping

# Defining evaluation metric
def AMSScore(s,b): 
    return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))

# loading data. I am assuming that this data already comes with no header, and that it is already pre-processed 
# Signal (s) is coded as 1 and background is coded as 0

data_train =  np.loadtxt( './data_train_normalized.csv', delimiter=',')

data_test =  np.loadtxt( './data_test_normalized.csv', delimiter=',')

# random seed to allow reproductivity
np.random.seed(42)
r = np.random.rand(data_train.shape[0])

# Generating training and validation samples to train the classifier.
# I am assuming 70/30 in training/validation samples 
# Put Y(truth), X(data), W(weight), and I(index) into their own arrays
print('Assigning data to numpy arrays.')

p = 0.9;
# First 90% are training
Y_train = data_train[:,32][r<p]
X_train = data_train[:,1:31][r<p]
W_train = data_train[:,31][r<p]
# Last 10% are validation
Y_valid = data_train[:,32][r>=p]
X_valid = data_train[:,1:31][r>=p]
W_valid = data_train[:,31][r>=p]

#%% 
##############################################################
########## Training the Classifier and select threshold ######
##############################################################


#Training mlp classifier
classifier = Sequential()
classifier.add(Dense(units = 400, input_dim=30, activation ='sigmoid'))
classifier.add(Dense(units = 300, activation ='sigmoid'))
classifier.add(Dense(units=200,activation = 'sigmoid'))
classifier.add(Dense(units=100,activation = 'sigmoid'))
classifier.add(Dense(units= 1, activation = 'sigmoid'))
    
classifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[metrics.mean_squared_error, 'accuracy'])

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

classifier.fit(X_train,Y_train, epochs = 30)
#sample_weight = W_train
#Testing the classifier
prob_predict_train = classifier.predict_proba(X_train)[:,0]
prob_predict_valid = classifier.predict_proba(X_valid)[:,0]


# decide the threshold
amstrain = [];
amsvalid  = [];
x_axis = [];

for i in range(1,100):
    pcut = np.percentile(prob_predict_train,i)
    #print(i)
    # This are the final signal and background predictions
    Yhat_train = prob_predict_train > pcut 
    Yhat_valid = prob_predict_valid > pcut

    # To calculate the AMS data, first get the true positives and true negatives
    # Scale the weights according to the r cutoff.
    Y_Positive_train = W_train*(Y_train==1.0)*(1.0/p)
    Y_Negative_train = W_train*(Y_train==0.0)*(1.0/p)
    
    Y_Positive_valid = W_valid*(Y_valid==1.0)*(1.0/(1-p))
    Y_Negative_valid = W_valid*(Y_valid==0.0)*(1.0/(1-p))

    # s and b for the training 
    s_train = sum ( Y_Positive_train*(Yhat_train==1.0) )
    b_train = sum ( Y_Negative_train*(Yhat_train==1.0) )
    
    s_valid = sum ( Y_Positive_valid*(Yhat_valid==1.0) )
    b_valid = sum ( Y_Negative_valid*(Yhat_valid==1.0) )
    
    amstrain.append(AMSScore(s_train,b_train))
    amsvalid.append(AMSScore(s_valid,b_valid))
    x_axis.append(i)

plt.plot(amsvalid)
plt.plot(amstrain)

# The objective is to maximize ams (in the validation set)
threshold = x_axis[amsvalid.index(max(amsvalid))]

#%%
#######################################################################
########### Generating the Evaluation File          ###################
#######################################################################

#Using the classifier to study tge data_set 

X_test = data_test[:,1:31]
IDs    = data_test[:,0]

prob_predict_test = classifier.predict_proba(X_test)[:,0]

testInversePermutation = prob_predict_test.argsort()

testPermutation = list(testInversePermutation)
for tI,tII in zip(range(len(testInversePermutation)),
                  testInversePermutation):
    testPermutation[tII] = tI


submission = np.array([[str(int(IDs[tI])),str(testPermutation[tI]+1),
                       's' if prob_predict_test[tI] >= threshold/100.0 else 'b'] 
            for tI in range(len(IDs))])
    
submission = np.append([['EventId','RankOrder','Class']],
                       submission, axis=0)

np.savetxt("./submission_MLP.csv",submission,fmt='%s',delimiter=',')





#%%
#######################################################################
### Doing a Nice Plot with predictions and real results          ######
### The weights don't enter here, so be carefull when analysing this ##
#######################################################################

# Creating a interesting plot where we have predictions and real data
Classifier_training_S = classifier.predict_proba(X_train[Y_train>0.5]).ravel()
Classifier_training_B = classifier.predict_proba(X_train[Y_train<0.5]).ravel()
  
c_max = max([Classifier_training_S.max(),Classifier_training_B.max()])
c_min = min([Classifier_training_S.min(),Classifier_training_B.min()])
  
# Get histograms of the classifiers
Histo_training_S = np.histogram(Classifier_training_S,bins=50,range=(c_min,c_max))
Histo_training_B = np.histogram(Classifier_training_B,bins=50,range=(c_min,c_max))
#Histo_testing_A = np.histogram(Classifier_testing_A,bins=50,range=(c_min,c_max))
  
# Lets get the min/max of the Histograms
AllHistos= [Histo_training_S,Histo_training_B]
h_max = max([histo[0].max() for histo in AllHistos])*1.2
h_min = 1.0
  
# Get the histogram properties (binning, widths, centers)
bin_edges = Histo_training_S[1]
bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
bin_widths = (bin_edges[1:] - bin_edges[:-1])
  
# Draw objects
ax1 = plt.subplot(111)
  
# Draw solid histograms for the training data
ax1.bar(bin_centers-bin_widths/2.,Histo_training_B[0],facecolor='red',linewidth=0,width=bin_widths,label='B (Train)',alpha=0.5)

ax1.bar(bin_centers-bin_widths/2.,Histo_training_S[0],bottom=Histo_training_B[0],facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
 
# Make a colorful backdrop to show the clasification regions in red and blue
ax1.axvspan(threshold, c_max, color='blue',alpha=0.08)
ax1.axvspan(c_min,threshold, color='red',alpha=0.08)
  
# Adjust the axis boundaries (just cosmetic)
ax1.axis([c_min, c_max, h_min, h_max])
  
#ax1.set_yscale('log')
    
# Make labels and title
plt.title("Higgs Kaggle Signal-Background Separation")
plt.xlabel("Probability Output (Gradient Boosting)")
plt.ylabel("Counts/Bin")
 
# Make legend with smalll font
legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
for alabel in legend.get_texts():
            alabel.set_fontsize('small')
  
# Save the result to png
#plt.savefig("naivebayesresults.png")
