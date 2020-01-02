import sys
import numpy 
import pandas 
import matplotlib
import seaborn
import scipy
import sklearn

print('python: {}'.format(sys.version))
print('numpy: {}'.format(numpy.__version__))
print('pandas: {}'.format(pandas.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('sklearn {}'.format(sklearn.__version__))


#import the necessary library 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading  the dataset
df=pd.read_csv('creditcard.csv')
print(df.columns)
print(df.shape)
print(df.describe())


#taking a small sample of the dataset

df=df.sample(frac=0.1,random_state=1)
print(df.shape)

#plot histogram of each parameter
df.hist(figsize=(20,20))
plt.show()


#deteremine the number of fraud cases in dataset
valid=df[df['Class']==0]
fraud=df[df['Class']==1]


print('fraud cases:{}'.format(len(fraud)))
print('valid cases:{}'.format(len(valid)))

#co-relation matrix
corrmat=df.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=9,square=True)
plt.show()

#get all the columns from the data frame

columns=df.columns.tolist()

#filter all the columns to remove the data we donnot want

columns =[c for c in columns if c not in ["Class"]] #removing the class column since it is the target

#STORING THE VARIABLE WE WILL BE PREDICTING ON
target="Class"
X=df[columns]
y=df[target]

print(X.shape)
print(y.shape)


#standardizing the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#USING THE LOGISTIC REGRESSION APPROACH.
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

#testing the model
y_pred=classifier.predict(X_test)

#classification report

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#using K-nn approach
from sklearn.neighbors import KNeighborsClassifier
Knn=KNeighborsClassifier()
Knn.fit(X_train,y_train)

#testing the model
y_predKn=classifier.predict(X_test)
print(classification_report(y_test,y_predKn))




#Random forest approach

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X_train,y_train)


