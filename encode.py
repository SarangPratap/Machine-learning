# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
df=pd.read_csv("Encoding Data.csv")
df.head(10)
df['bin_1'] = df['bin_1'].apply(lambda x: 1 if x=='T' else (0 if x=='F' else None))
df['bin_2'] = df['bin_2'].apply(lambda x: 1 if x=='Y' else (0 if x=='N' else None))
sns.countplot(df['bin_1'])
sns.countplot(df['bin_2'])
dfcopy=df
#label encoding
from sklearn.preprocessing import LabelEncoder  
le=LabelEncoder()
dfcopy['ord_2'] = le.fit_transform(dfcopy['ord_2'])
sns.set(style="darkgrid")
sns.countplot(dfcopy['ord_2'])
"""
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
enc=enc.fit_transform(df[['nom_0']]).toarray()
encoded_colm=pd.DataFrame(enc)
df=pd.concat([df,encoded_colm],axis=1)
df=df.drop(['nom_0'],axis=1)
df.head(10)"""
"""
df = pd.get_dummies(df, prefix = ['nom_0'], columns = ['nom_0'])
df.head(10)
"""
fe = df.groupby('nom_0').size()/len(df)
df.loc[:, "{}_freq_encode".format('nom_0')] = df['nom_0'].map(fe)    
df=df.drop(['nom_0'],axis=1)
fe.plot.bar(stacked=True)
df.head(10)

dtype='np.int64'



from sklearn.preprocessing import OrdinalEncoder
ord1=OrdinalEncoder()
ord1.fit([df['ord_2']])
df["ord_2"]=ord1.fit_transform(df[["ord_2"]])
df.head(10)
dnew=df.copy()
temp_dict ={'Cold':1, 'Warm':2, 'Hot':3}
dnew['Ord_2_encod']=dnew.ord_2.map(temp_dict)
dnew=dnew.drop(['ord_2'],axis=1)




from category_encoders import BinaryEncoder
encoder=BinaryEncoder(cols=['ord_2'])
newdata=encoder.fit_transform(df['ord_2'])
df=pd.concat([df,newdata],axis=1)
df=df.drop(['ord_2'],axis=1)
df.head(10)


from sklearn.feature_extraction import FeatureHasher
h=FeatureHasher(n_features=3 ,input_type='string')
hashed_Feature=h.fit_transform(df['nom_0'])
hashed_Feature=hashed_Feature.toarray()
df=pd.concat([df,pd.DataFrame(hashed_Feature)],axis=1)
df.head(10)


df.insert(6, "Target", [0,1,1,0,0,1,0,0,0,1], True) 


mean = train['target'].mean()
agg = train.groupby(col)['target'].agg(['count','mean'])
counts = agg['count']
means = agg['mean']
weight = 100   
smooth = ((counts * means) + (weight * mean)) / (counts + weight)
train.loc[:,"{}_mean_encode".format(col)] = train[col].map(smooth)



