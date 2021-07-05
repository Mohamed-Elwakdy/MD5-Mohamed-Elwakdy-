

# Lehner Investments - Assessment
 

<br>

## Introduction 

<br>

#### This dataset pull from The Wharton School, University of Pennsylvania. 

#### I used Python 3 to work on this Assesment. I built two predictive models using XGBoost and Random Forest. XGBoost and Random forest are the best two machine learning algorithms that I used for building predictive models. XGboost can handle the missing values and Random Forest can be used to get a low variance in addion to this algorithm is used to find out the most important independent variables and can be used  with subsampling techniques to deal with a big data.   

#### The best perfomance of both models I get using those two predictive models when I increased the number of trees 'n_estimators' and 'max_depth' hyperparameters where I got the highest train_scores and test_scores and lowest mse_train and mse_test.  
 
<br>

## Import Packages 

```python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import pandas_profiling
from pandas_profiling import ProfileReport 
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import confusion_matrix
from xgboost import plot_tree
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

```
<br>

## Read Data file

#### The head () function is used to get the first n rows.

```python
df = pd.read_csv('Dataset_Januray_2005.csv',na_values = ' ') 
```

<p align="center">
  <img width="850" height="250" src="https://user-images.githubusercontent.com/61699200/124518680-0ead7580-ddb5-11eb-8532-bed8698e3c32.jpg">
</p>

<br>

## Field names and Description
#### We can show the field names(header) by showing the data of the 1st row (0th index).

<br>

|    Field Name        |        Description                |
|:-: | ------------------ |
|PERMNO | Unique stock (share class) level identifier|
|Date | Weekly basis |
|n | Number of shares (stocks)|
|RET | Returns that the investors generate out of the stock market|
|B_MKT	|Beta on MKT (levered market beta)|
|ALPHA | The excess return on an investment after adjusting for market-related volatility and random fluctuations|
|IVOL	| Idiosyncratic Volatility (effective inflation and market hedge)|
|TVOL	 | Total Volatility (trading volume, which refers to the number of share s traded in the stock)|
|R2	| R-Squared|
|EXRET	| Excess Return from Risk Model|

<br>

## Data Cleaning 

```python
# Convert Percentage string to float in dataframe

list1 = ["ivol","tvol","R2","exret","RET"]

for item in list1: 
    df[item] = df[item].str.rstrip('%').astype('float')/ 100.0

# Replace all zeros in dataframe with NaN
df = df.replace(0, np.nan)

# Drop all rows which contains 'NAN' in the dependent variable 'RET'
df = df.dropna(subset = ['RET'])

```

<p align="center">
  <img width="780" height="200" src="https://user-images.githubusercontent.com/61699200/124521776-c09d6f80-ddbe-11eb-8006-fb22059be3ff.jpg">
</p>

<br>

#### Extract the day information form 'DATE' independent variable because this feature is an important feature and has an effect on the dependent variable 'RET' and improve the performance of the predictive model. 

```python

df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['Day'] =  df['DATE'].dt.day

# Remove 'DATE' column from dataframe.  
df = df.drop(['DATE'], axis = 1)

```
<br>

#### Apply normalization techniques to treate the negative values with changing the location for the 'RET' independent vaiable to be the last column in the dataframe 
```python
for column in df.columns:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

df = df[['PERMNO','n','b_mkt','alpha','ivol','tvol','R2','exret','Day','RET']]
```
<br>

<p align="center">
  <img width="800" height="200" src="https://user-images.githubusercontent.com/61699200/124523048-4de2c300-ddc3-11eb-915f-6a2ad007b7b4.jpg">
</p>

<br>

## Check the correlation between the variables in the dataset

#### Tvol (Total Volatility), Ivol (Idiosyncratic Volatility) seems have high predictive power but highly correlated "0.99". 
#### A lots for the features are not heavily correlated

```python
feat_cols1 = []
for col in df1.columns:
    #print(col)
    feat_cols1.append (col)
feat_cols1.pop()
print (feat_cols1)   

length_mat = len(feat_cols1)
corr_ = df1[feat_cols1].corr()
print (corr_)

corr_thres = 0.8

for row in list(range(length_mat)):
    for col in list(range(row)):
        corr_val = corr_.iloc[row,col]
        if corr_val > corr_thres:
            print(corr_.index[row],'is correlated with:',corr_.index[col],'with correlation value of',corr_val)
```

<br>

<p align="center">
  <img width="800" height="350" src="https://user-images.githubusercontent.com/61699200/124523828-a8c9e980-ddc6-11eb-9cad-3a334d05a07f.jpg">
</p>

