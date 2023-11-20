import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , accuracy_score , precision_recall_fscore_support
pd.set_option('display.max_columns', None)

df = pd.read_csv("IRIS.csv")

print(df)
print(df.info())
print(df.describe())
print("missing values count : " , df.isna().sum())

#Missing values 
missingno.matrix(df)
plt.title("missing values")
plt.show()
plt.figure()

for col in df.columns:
    if col != "species":
        sns.scatterplot(df[col])
        plt.title(f'Scatter plot : {col}')
        plt.show()
        plt.figure()

for col in df.columns:
    if col != "species":
        sns.boxplot(df[col])
        plt.title(f'Box plot : {col}')
        plt.show()
        plt.figure()

sns.countplot(df['species'])
plt.title("count of species")
plt.show()
plt.figure()

for i, column in enumerate(df.columns):
    for j, column2 in enumerate(df.columns):
        if j > i and column != column2:
            sns.scatterplot(x=column, y=column2, data=df, hue="species")
            plt.title(f'Scatter Plot: {column} vs {column2}')
            plt.xlabel(column)
            plt.ylabel(column2)
            plt.show()
            plt.figure()

#mapping categories to numbers for processing
df['species'] = df['species'].astype("category")
mapping = {'Iris-setosa': 0 , 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
df['species'] = df['species'].map(mapping)

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

#replacing outliers of sepal_width with median
iqr = df["sepal_width"].quantile(0.75) - df["sepal_width"].quantile(0.25)
lower_bound = df["sepal_width"].quantile(0.25) - (iqr * 1.5)
upper_bound = df["sepal_width"].quantile(0.75) + (iqr * 1.5)

mean_sepal_width = df[(df["sepal_width"] >= lower_bound) & (df["sepal_width"] <= upper_bound)]["sepal_width"].median()
df.loc[df["sepal_width"] < lower_bound, "sepal_width"] = mean_sepal_width
df.loc[df["sepal_width"] > upper_bound, "sepal_width"] = mean_sepal_width

#after removing outliers
sns.boxplot(df['sepal_width'])
plt.title("sepal_length after replacing outliers")
plt.show()
plt.figure()

x = df.drop('species' , axis = 1)
y = df['species']

x = StandardScaler().fit_transform(x)

X_train , X_test , Y_train  , Y_test = train_test_split(x,y, train_size = 0.8)

params = {
    'n_estimators': [10,50],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

gscv = GridSearchCV(RandomForestClassifier() , param_grid= params , cv = 2, scoring='accuracy')
gscv.fit(X_train , Y_train)

best_model = gscv.best_estimator_
best_model.fit(X_train, Y_train)

y_pred = best_model.predict(X_test)

sns.heatmap(confusion_matrix(Y_test , y_pred) , annot = True , xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], yticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Accuracy of model  :" , accuracy_score(Y_test , y_pred))

precision, recall, fscore, _ = precision_recall_fscore_support(Y_test, y_pred, average='macro')

print("precision of model  :" , precision)
print("recall of model  :" , recall)
print("fscore of model  :" , fscore)


