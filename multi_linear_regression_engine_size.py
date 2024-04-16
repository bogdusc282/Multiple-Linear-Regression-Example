import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
 
inputdata = pd.read_csv("car details v4.csv")
 
print(inputdata)
#The following line is necessary to drop the NaN lines from the dataset:
inputdata = inputdata.dropna()
#The following line removes the text 'cc' from the column 'Engine'
inputdata['Engine'] = inputdata['Engine'].str.replace('cc', '')
#The following line ensures that the values of column 'Engine' corresponds to 64-bit integers
inputdata['Engine'] = inputdata['Engine'].astype('int64')
#print(inputdata)
#y is the predicted value, corresponding to the Engine size
y = inputdata.iloc[:, 11].values
#X is the predictor, corresponding to car's vehicle Length, Seating Capacity and Fuel Tank Capacity
X = inputdata.iloc[:,[15, 18, 19]].values#Length, Seating Capacity, Fuel Tank Capacity 
print(X)
print(y)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/4, random_state=33)
 
ml_model = LinearRegression()
ml_model.fit(X_train, y_train)
 
y_pred = ml_model.predict(X_test)
 
d = y_pred - y_test
print(d)
#Calculate Mean Squared Error (MSE)
mse = np.mean(d**2)
print(f"mse = {mse}")
#Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"rmse = {rmse}")
 
#Plot the predicted and the actual values for first 50 data points 
plt.plot(range(0,50),y_test[:50],c="blue", label = "real value (y_test)")
plt.plot(range(0,50),y_pred[:50],c="red", label = "predicted value (y_pred)")
print(y_test)
print(y_pred)
plt.grid(True)
plt.xlabel('Data points')
plt.ylabel('Engine size [cc]')
plt.title('Multiple Linear Regression prediction')
plt.legend(loc="upper right")
plt.show()
