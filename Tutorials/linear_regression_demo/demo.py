import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('challenge_dataset.txt', header=None)
x_values = dataframe[0]
y_values = dataframe[1]


#train model on data
reg = linear_model.LinearRegression()
reg.fit(x_values.values.reshape(-1,1), y_values)

plt.scatter(x_values,y_values)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, reg.predict(x_values.values.reshape(-1,1)))
plt.show()
