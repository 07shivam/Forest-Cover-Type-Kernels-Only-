# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:50:45 2019

@author: Shivam Bhargava
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from goto import with_goto
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

df_train=pd.read_csv("file:///C:/Users/Shivam Bhargava/Desktop/Forest/train.csv")
x_test=pd.read_csv("file:///C:/Users/Shivam Bhargava/Desktop/Forest/test.csv")
y_train = df_train.iloc[:,55].values
x_train = df_train.drop('Cover_Type', axis=1)
print("\nData of testting :\n",x_test.info)
print("\nData of training :\n",x_train.info)
print("\nData of target :\n",df_train['Cover_Type'])

print("\nShape of x :\n",x_train.shape)
print("\nShape of y :\n",y_train.shape)

print("\nPrinting Unique Values of Y :\n",np.unique(y_train))
print("\nTotal No of Unique Values :\n ",np.unique(y_train).sum())
print("\nNumber of Attritube :\n",x_train.shape[1])
print("\nNumber of Instance :\n",x_train.shape[0])

print("\nChecking Null Values :\n",df_train.isnull().sum())
#Scaling of data
print("\nScaling of Data with MinMaxScaler :\n")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

print("\nStandardizing of Data :\n")
#standardizing
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,test_size=0.3,random_state=None)
print("\nShape of x_train :\n",x_train.shape)
print("\nShape of x_test :\n",x_test.shape)
print("\nShape of y_train :\n",y_train.shape)
print("\nShape of y_test :\n",y_test.shape)

ch='y'
while(ch=='y'):
    s= int(input("1. SVM \n2. Logistic Regression \n3. KNN \n4.Random Forest With Features Importance \n5.DecisionTreeClassifier"))
    
    if s==1:
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(x_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(x_test)
        
        print("\nPrediction\n")
        df=pd.DataFrame(df_train['Id'],y_pred)
        print(df)
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print('\nMisclassified Samples: %d'%(y_test!=y_pred).sum())
        from sklearn.metrics import accuracy_score
        print('\nAccuracy with Accuracy_Score:%.2f'%accuracy_score(y_test,y_pred))
        from sklearn.metrics import mean_squared_error
        #besthyper parameter
        from math import sqrt
        rms = sqrt(mean_squared_error(y_test, y_pred))
        print('\nAccuracy With Mean Square Error : ',100 - rms)
                
        error = 1000
        ind = 0
        l = ['linear','rbf']
        for i in l:
            svmC =SVC(kernel = i, random_state = 0)
            svmC.fit(x_train,y_train)
            y_pred=svmC.predict(x_test)
            rms = sqrt(mean_squared_error(y_test, y_pred))
            if rms < error:
                error = rms
                ind = i        
                #print best value of estimator
        print(100 - error, " Accuracy With Best Value Of Estimator Is ", ind)
        from sklearn.metrics import zero_one_loss
       # print("\nZero - One Loss : " ,zero_one_loss(y_test, y_pred))
        print("Press y to continue and n to quit")
        ch=(input())
           
    if s==2:
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(x_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(x_test)
        
        print("\nPrediction\n")
        df=pd.DataFrame(df_train['Id'],y_pred)
        print(df)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print('\nMisclassified Samples: %d'%(y_test!=y_pred).sum())
        from sklearn.metrics import accuracy_score
        print('\nAccuracy with Accuracy_Score:%.2f'%accuracy_score(y_test,y_pred))
        #besthyper parameter
        from math import sqrt
        rms = sqrt(mean_squared_error(y_test, y_pred))
        print('\nAccuracy With Mean Square Error : ',100 - rms)    
        
        error = 1000
        ind = 0
        l=['l1','l2']
        for i in l:
            logC =LogisticRegression(penalty= i,random_state = 0)
            logC.fit(x_train,y_train)
            y_pred=svmC.predict(x_test)
            rms = sqrt(mean_squared_error(y_test, y_pred))
            if rms < error:
                error = rms
                ind = i
                #print best value of estimator
        print(100 - error, " Accuracy With Best Value Of Estimator Is ", ind)
        from sklearn.metrics import zero_one_loss
       # print("\nZero - One Loss : " ,zero_one_loss(y_test, y_pred))
        print("Press y to continue and n to quit")
        ch=(input())
        
    if s==3:            
    # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(x_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(x_test)  
        
        print("\nPrediction\n")
        df=pd.DataFrame(df_train['Id'],y_pred)
        print(df)
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print('\nMisclassified Samples: %d'%(y_test!=y_pred).sum())
        from sklearn.metrics import accuracy_score
        print('\nAccuracy with Accuracy_Score:%.2f'%accuracy_score(y_test,y_pred))
        #besthyper parameter
        from math import sqrt
        rms = sqrt(mean_squared_error(y_test, y_pred))
        print('\nAccuracy With Mean Square Error : ',100 - rms)    
        
        error = 1000
        ind = 0
        for i in range(1,10):
            knnC =KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
            knnC.fit(x_train,y_train)
            y_pred=svmC.predict(x_test)
            rms = sqrt(mean_squared_error(y_test, y_pred))
            if rms < error:
                error = rms
                ind = i
                #print best value of estimator
        print(100 - error, " Accuracy With Best Value Of Estimator Is ", ind)
        from sklearn.metrics import zero_one_loss
       # print("\nZero - One Loss : " ,zero_one_loss(y_test, y_pred))
        print("Press y to continue and n to quit")
        ch=(input())
 
    if s==4:
        from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
        forest = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=0)
        forest.fit(x_train,y_train)
        y_pred=forest.predict(x_test)
        
        print("\nPrediction\n")
        df=pd.DataFrame(df_train['Id'],y_pred)
        print(df)
        print("\nFeature Importance",forest.feature_importances_)
        print('\nMisclassified Samples: %d'%(y_test!=y_pred).sum())
        from sklearn.metrics import accuracy_score
        print('\nAccuracy with Accuracy_Score:%.2f'%accuracy_score(y_test,y_pred))
        from math import sqrt
        rms = sqrt(mean_squared_error(y_test, y_pred))
        print('\nAccuracy With Mean Square Error : ',100 - rms)
        
        error = 1000
        ind = 0
        for i in range(1,10):
            forest = RandomForestClassifier(criterion='entropy',n_estimators=i,random_state=0)
            forest.fit(x_train,y_train)
            y_pred=forest.predict(x_test)
            rms = sqrt(mean_squared_error(y_test, y_pred))
            if rms < error:
                error = rms
                ind = i
                #print best value of estimator
        print(100 - error, " Accuracy With Best Value Of Estimator Is ", ind)
        from sklearn.metrics import zero_one_loss
        #print("\nZero - One Loss : " ,zero_one_loss(y_test, y_pred))
       
        print("Press y to continue and n to quit")
        ch=(input())
    
    if s==5:
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
        tree.fit(x_train,y_train)
        y_pred=tree.predict(x_test)
        
        print("\nPrediction\n")
        df=pd.DataFrame(df_train['Id'],y_pred)
        print(df)
        
        print('misclassified samples: %d'%(y_test!=y_pred).sum())
        from sklearn.metrics import accuracy_score
        print('Accuracy:%.2f'%accuracy_score(y_test,y_pred))
        from math import sqrt
        rms = sqrt(mean_squared_error(y_test, y_pred))
        print('\nAccuracy With Mean Square Error : ',100 - rms)
        
        error = 1000
        ind = 0
        for i in range(1,10):
            tree = DecisionTreeClassifier(criterion='entropy',max_depth=i,random_state=0)
            tree.fit(x_train,y_train)
            y_pred=tree.predict(x_test)
            rms = sqrt(mean_squared_error(y_test, y_pred))
            if rms < error:
                error = rms
                ind = i
                #print best value of estimator
        print(100 - error, " Accuracy With Best Value Of Estimator Is ", ind)
        from sklearn.metrics import zero_one_loss
        #print("\nZero - One Loss : " ,zero_one_loss(y_test, y_pred))
        
        print("Press y to continue and n to quit")
        ch=(input())

    

  
from sklearn.decomposition import PCA       
pca = PCA(n_components=3)
train_pca = pca.fit_transform(x_train)
print('\nRepresentation of dataset in 3 dimensions:\n')
print(train_pca)

for i in range(1,10):
  df_train.loc[i].plot()  

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier 
etc = ExtraTreesClassifier(n_estimators=350)  
etc.fit(x_train, y_train)
#sub = pd.DataFrame({"Id": x_test['Id'],"Cover_Type": etc.predict(x_test)})
y_pred=etc.predict(x_test)
df=pd.DataFrame(df_train['Id'],y_pred)
print(df)


'''sub.to_csv("etc.csv", index=False)
one=pd.read_csv("etc.csv")
print(one)'''