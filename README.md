# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:ABIJITH SHAAN S 
RegisterNumber:212223080002
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset

![Screenshot 2024-02-29 214206](https://github.com/Shaan2803/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568486/62ea0e41-3f1c-49ab-917e-ebe0934277e7)

df.head()

![Screenshot 2024-02-29 214215](https://github.com/Shaan2803/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568486/39781399-044d-432d-b627-3f23aa66759e)

f.tail()

![Screenshot 2024-02-29 214221](https://github.com/Shaan2803/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568486/58407250-3d11-4e61-a332-1b8e410795bf)

X and Y values

![Screenshot 2024-02-29 214241](https://github.com/Shaan2803/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568486/e63f2ffe-1143-480f-a7da-cbd16e3ddd5f)

Predication values of X and Y

![Screenshot 2024-02-29 214251](https://github.com/Shaan2803/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568486/04956a7b-e72e-4052-a4ab-5237bee3689f)

MSE,MAE and RMSE

![Screenshot 2024-02-29 214257](https://github.com/Shaan2803/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568486/444ba446-7d2c-4790-ac76-5f58f61afc77)

Training Set

![Screenshot 2024-02-29 214138](https://github.com/Shaan2803/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568486/1bd4e777-b95e-4976-a262-06eb30567172)

Testing Set

![Screenshot 2024-02-29 214147](https://github.com/Shaan2803/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568486/a9bcce77-f50d-4d38-8e3e-860b196234b1)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
