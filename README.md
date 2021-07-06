

# Lehner Investments - Assessment
 

<br>

## Introduction 

<br>

#### This dataset pull from The Wharton School, University of Pennsylvania. 

#### I used Python 3 to work on this Assesment. I built two predictive models using XGBoost and Random Forest. XGBoost and Random forest are the best two machine learning algorithms that I used for building predictive models. XGboost can handle the missing values and Random Forest can be used to get a low variance in addion to this algorithm is used to find out the most important independent variables and can be used with subsampling techniques to deal with a big data.   

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
  <img width="730" height="320" src="https://user-images.githubusercontent.com/61699200/124523828-a8c9e980-ddc6-11eb-9cad-3a334d05a07f.jpg">
</p>

<br>

## Split the dataset into training and test datasets 


```python

def Split_Training_Test_fun (df_train_test):
    
    train_set = df_train_test.sample(frac=0.80, random_state=0)
    test_set = df_train_test.drop(train_set.index)
    train_set_RET = train_set.pop('RET')
    test_set_RET = test_set.pop('RET')
    
    # Convert train_set, test_set,train_set_labels and test_set_labels to float32 
    train_set = np.nan_to_num(train_set.astype(np.float32))
    train_set_RET = np.nan_to_num(train_set_RET.astype(np.float32))
    test_set = np.nan_to_num(test_set.astype(np.float32))
    test_set_RET = np.nan_to_num(test_set_RET.astype(np.float32))
    
    return train_set,test_set,train_set_RET,test_set_RET

train_set,test_set,train_set_RET,test_set_RET = Split_Training_Test_fun (df)

```
<br>

## XGBoost Algorithm

#### Used the XGBoost Algorithm for training and evaluate the model with different numbers of trees (from 6 to 20 trees) with calculating the mse_train, mse_test, mse_train, mse_test, RMSE_Train and RMSE_Test 

<br>

```python

model = XGBRegressor(n_estimators = 6)

#def Train_Evaluate_Model_Fun (model,train_set, test_set, train_set_RET, test_set_RET):
    
for iter in range(6, 21, 1):
    
    num_trees.append(iter)
    
    model.fit(train_set, train_set_RET)
    y_train_predicted = model.predict(train_set)
    train_score =  model.score(train_set, train_set_RET)
    train_scores.append(train_score)
    
    y_test_predicted = model.predict(test_set)
    test_score =  model.score(test_set, test_set_RET)
    test_scores.append(test_score)    

    mse_train = mean_squared_error(train_set_RET, y_train_predicted)
    mse_train1.append(mse_train)
    RMSE_Train = np.sqrt(mse_train)
    RMSE_Train1.append (RMSE_Train)
    
    mse_test = mean_squared_error(test_set_RET, y_test_predicted)
    mse_test1.append(mse_test)
    RMSE_Test = np.sqrt(mse_test)
    RMSE_Test1.append (RMSE_Test)
    #print('>%d, train: %.3f, test: %.3f' % (i, train_score, test_score))
    #print("Iteration: {} Train mse: {} Test mse: {}".format(iter, mse_train, mse_test))
    model.n_estimators += 1
    #return train_scores, test_scores,mse_train1,mse_test1
    
    #train_scores, test_scores,mse_train1,mse_test1 = Train_Evaluate_Model_Fun (model,train_set,test_set,train_set_RET,test_set_RET)
    
```

#### Based on the "train_scores" and "test_scores", the performance of the model is improved when the number of trees are increased so the highest "train_scores" and "test_scores" I got "> 99%" when the number of trees > 9. Although the y-axis is so small, meaning the deviations shown in the plot may not be very significant regarding the underfitting and overfitting as they are so small.  I am looking to find out the best hyperparameters values to get the best performance of the model. Based on Figure 1, what I can start to visually see that there is underfitting when the number of trees <=7, but there is no overfitting.

#### For more information about the mse_train, mse_test, mse_train, mse_test, RMSE_Train and RMSE_Test values, plaese see the Jupyter Notebook  

```python

  def plot_fun (num_trees,mse_train1, mse_test1):
    
    pyplot.plot(num_trees, mse_train1, marker='.', label= 'MSE on Train Data')
    pyplot.plot(num_trees, mse_test1, marker='.', label= 'MSE on Test Data')

    # axis labels
    pyplot.xlabel('no. of trees')
    pyplot.ylabel('Mean Squared Error')
    
    # show the legend
    pyplot.legend()
    
    # show the plot
    pyplot.show()

plot_fun(num_trees,mse_train1, mse_test1)
```
<br>

<p align="center">
  <img width="600" height="370" src="https://user-images.githubusercontent.com/61699200/124529921-88575a80-ddd9-11eb-8605-6820fba32082.jpg">
</p>

<p align="center">
     Figure 1 XGBOoost Algorithm - Mean Squared Error Against Number of Trees (n_estimators has values from 6 to 20)
</p>

<br>

#### Finding the importantance variables

<br>


```python

# Get feature importance in xgboost and sort it with descending.

sorted_idx = np.argsort(model.feature_importances_)[::-1]

# print all sorted importances and the name of columns together as lists

for index in sorted_idx:
    print([df.columns[index], model.feature_importances_[index]])

# Plot the importances with XGboost built-in function

# Although the 'Day' independent variable has small score, it is the third feature in 
# the importance and has an effect on the performance of the model 

plot_importance(model) # Plot importance based on fitted trees.
plt.show()

```
<p align="center">
  <img width="600" height="370" src="https://user-images.githubusercontent.com/61699200/124533106-be97d880-dddf-11eb-8201-0b6c1999b157.jpg">
</p>

<p align="center">
  <img width="600" height="370" src="https://user-images.githubusercontent.com/61699200/124533156-d2dbd580-dddf-11eb-8a06-860d6f7bc208.jpg">
</p>



  
  
  
