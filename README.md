

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


