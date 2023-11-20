import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option('display.max_columns', None)

df = pd.read_csv("carsales.csv" , encoding='ISO-8859-1')

print(df.info())
print(df.describe())
print("missing values count : " , df.isna().sum())

#Missing values 
missingno.matrix(df)
plt.title("missing values")
plt.show()
plt.figure()

df.drop(["customer name", "customer e-mail" , "country"], axis = 1, inplace= True)

#gender count
df['gender'].value_counts().plot(kind='bar')
plt.xticks(ticks=[0, 1], labels=['female', 'male'] , rotation = 90)
plt.title("Count of gender")
plt.show()
plt.figure()

float_cols = ["age" , "annual Salary" , "credit card debt" , 'net worth' , 'car purchase amount']

for col in float_cols:
    sns.boxplot(df[col])
    plt.title(f'Box plot : {col}')
    plt.show()
    plt.figure()

for i, column in enumerate(float_cols):
    for j, column2 in enumerate(float_cols):
        if j > i and column != column2:
            sns.scatterplot(x=column, y=column2, data=df, hue="car purchase amount" , palette="coolwarm")
            plt.title(f'Scatter Plot: {column} vs {column2}')
            plt.xlabel(column)
            plt.ylabel(column2)
            plt.show()
            plt.figure()

#correlation heatmap
plt.figure(figsize = (16,7))
sns.heatmap(df.corr() , cmap= "inferno" , annot= True)
plt.title("Correlation matrix")
plt.show()
plt.figure()

#pairplot of all columns
sns.pairplot(data= df)
plt.show()
plt.figure()

#removing outliers, since there are only 15
for col in float_cols:
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower_bound = df[col].quantile(0.25) - (iqr * 1.5)
    upper_bound = df[col].quantile(0.75) + (iqr * 1.5)
    df.loc[df[col] < lower_bound, col] = np.NaN
    df.loc[df[col] > upper_bound, col] = np.NaN
    df.dropna(inplace=True, axis = 0)

print(df.shape)

for col in float_cols:
    sns.boxplot(df[col])
    plt.title(f'Box plot (after removing outliers): {col}')
    plt.show()
    plt.figure()

x = df.drop("car purchase amount" , axis = 1)
y = df['car purchase amount']

x = StandardScaler().fit_transform(x)

X_train , X_test , Y_train  , Y_test = train_test_split(x,y, train_size = 0.8)

params = {
    'n_estimators': [10,50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

gscv = GridSearchCV(RandomForestRegressor() , param_grid= params , cv = 2, scoring='accuracy')
gscv.fit(X_train , Y_train)

best_model = gscv.best_estimator_

best_model.fit(X_train, Y_train)

y_pred = best_model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("mean square error of model : " , mse)
print("coefficient of determination : " , r2)
