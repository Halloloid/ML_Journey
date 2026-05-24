# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

data = "https://github.com/ageron/data/raw/main/lifesat/lifesat.csv"
lifesat = pd.read_csv(data)

# print(lifesat)

lifesat.plot(kind="scatter",x="GDP per capita (USD)",y="Life satisfaction",grid=True)
plt.axis((23_500,62_500,4,9))
plt.show()

X = lifesat[["GDP per capita (USD)"]].values
Y = lifesat[["Life satisfaction"]].values

model1 = LinearRegression()
model2 = KNeighborsRegressor()

model1.fit(X,Y)
model2.fit(X,Y)

X_new = [[37_655.2]]

print(model1.predict(X_new))
print(model2.predict(X_new))