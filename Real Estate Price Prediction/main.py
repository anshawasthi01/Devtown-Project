## importing

from pickle import TRUE
from pkgutil import ImpImporter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import alphanums
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump,load
np.random.seed(42)

## Initialising 

house=pd.read_csv("dat.csv")

## Just analysing data

# print(house.head())
# print(house.info())
# print(house['CHAS'].value_counts())
# print(house.describe())
# house.hist(bins=50, figsize=(20,15))
# plt.show()

## Test-Train Splitting manually

# shuff=np.random.permutation(len(house))
# print(shuff)
# set_size=int(len(house)*0.2)
# test_ind=shuff[:set_size]
# train_ind=shuff[set_size:]
# train_set=house.iloc[train_ind]
# test_set=house.iloc[test_ind]
# print(len(test_set),len(train_set))

## Test-Train Splitting from sklearn

train_set, test_set=train_test_split(house,test_size=0.2, random_state=42)
# print(len(test_set),len(train_set))

## Stratified_Splitting

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(house, house['CHAS']):
    strat_train_set=house.loc[train_index]
    strat_test_set=house.loc[test_index]
house=strat_train_set.copy()

## Finding Correlations

corr_matrix=house.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False))
attributes=['MEDV','RM','ZN','LSTAT']
scatter_matrix(house[attributes],figsize=(12,8))
house.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)
# plt.show()

## Attributes Combine

# house['TPM']=house['TAX']/house['RM']
# corr_matrix=house.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False))
# house.plot(kind='scatter',x='RM',y='MEDV',alpha=0.8)
# plt.show()

## Spltting data and label

house=strat_train_set.drop("MEDV",axis=1)
house_label=strat_train_set['MEDV'].copy()

## Missing Attributes Handling Manually

# house.dropna(subset=['RM'],inplace=TRUE) #missing data points gone
# house.drop('RM',axis=1,inplace=TRUE) #RM attribute removed
# median=house['RM'].median()
# house['RM'].fillna(median,inplace=TRUE) #set value with median

## Missing Attributes Handling From Sklearn

# imputer=SimpleImputer(strategy='median')
# imputer.fit(house)
# print(imputer.statistics_)
# D=imputer.transform(house)
# house_tr=pd.DataFrame(D,columns=house.columns)

## Pipelining

pipl=Pipeline([('imputer',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())])
house_tr_num=pipl.fit_transform(house)

## Linear Regression

model=LinearRegression()
model.fit(house_tr_num,house_label)

## Some Testing

some_data=house.iloc[:5]
some_label=house_label.iloc[:5]
fin_data=pipl.transform(some_data)
# print(model.predict(fin_data))
# print(list(some_label))

## Evaluation

house_pred=model.predict(house_tr_num)
mse=mean_squared_error(house_label,house_pred)
rmse=np.sqrt(mse)
# print(mse)
# print(rmse)

## Cross Validation

scores=cross_val_score(model,house_tr_num,house_label,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
# print(rmse_scores)


## Decision Tree Regressor

model=DecisionTreeRegressor()
model.fit(house_tr_num,house_label)

## Some Testing

some_data=house.iloc[:5]
some_label=house_label.iloc[:5]
fin_data=pipl.transform(some_data)
# print(model.predict(fin_data))
# print(list(some_label))

## Evaluation

house_pred=model.predict(house_tr_num)
mse=mean_squared_error(house_label,house_pred)
rmse=np.sqrt(mse)
# print(mse)
# print(rmse)

## Cross Validation

scores=cross_val_score(model,house_tr_num,house_label,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
# print(rmse_scores)

## Random Forest Regressor

model=RandomForestRegressor()
model.fit(house_tr_num,house_label)

## Some Testing

some_data=house.iloc[:5]
some_label=house_label.iloc[:5]
fin_data=pipl.transform(some_data)
# print(model.predict(fin_data))
# print(list(some_label))

## Evaluation

house_pred=model.predict(house_tr_num)
mse=mean_squared_error(house_label,house_pred)
rmse=np.sqrt(mse)
# print(mse)
# print(rmse)

## Cross Validation

scores=cross_val_score(model,house_tr_num,house_label,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
# print(rmse_scores)

## Finalising

dump(model,'Realestate.joblib')