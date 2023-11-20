import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split , GridSearchCV 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score , precision_recall_fscore_support
pd.set_option('display.max_columns', None)

df = pd.read_csv("titanic.csv")

#analyzing the dataframe
print(df.info())
print(df.describe())

#removing passanger name, id, cabin and ticket since they do not help with predicting values
df.drop(["PassengerId" , "Name" , "Cabin" , "Ticket"] , axis=1 , inplace= True)

print(df)

#visualizing missing values
missingno.matrix(df)
plt.title("missing values")
plt.show()
plt.figure()

#number of missing values
print(df.isna().sum())

#visualizing age and fare columns 
sns.displot(df["Age"])
plt.title("Histogram of Ages")

plt.show()
plt.figure()

sns.boxplot(df["Age"])
plt.title("Boxplot of Ages")

plt.show()
plt.figure()

sns.displot(df["Fare"], bins = 20)
plt.title("Histogram of Fares")

plt.show()
plt.figure()


sns.boxplot(df["Fare"])
plt.title("Boxplot of Fares")

plt.show()
plt.figure()

#count of genders
plt.title("Count of Genders")
sns.countplot(x = "Sex" , data = df)

plt.show()
plt.figure()

#age relative to sex
plt.title("Age vs Sex")
sns.boxplot(x='Sex', y = 'Age',data= df)

plt.show()
plt.figure()

#count of embarked
plt.title("Count of Embarked") 
sns.countplot(x = "Embarked" , data= df)

plt.show()
plt.figure()

#count of pclass
plt.title("Count of Pclass")
sns.countplot(x = "Pclass" , data = df)

plt.show()
plt.figure()

#fare relative to pclass
plt.title("Fare vs Pclass")
sns.boxplot(x = "Pclass" , y = "Fare" , data=df)

plt.show()
plt.figure()

#fare relative to embarked
plt.title("Fare vs Embarked")
sns.boxplot(x = "Embarked" , y = "Fare" , data=df)

plt.show()
plt.figure()

#survival rate relative to embarked
plt.title("Survival rate with respect to embarked")
sns.countplot(x = "Embarked" , data = df, hue="Survived")

plt.show()
plt.figure()

#survival rate relative to pclass
plt.title("Survival rate with respect to Pclass")
sns.countplot(x = "Pclass" , data = df, hue="Survived")

plt.show()
plt.figure()

#converting embarked and sex to categorical columns to calculate correlation
mapping1 = {'S': 1, 'Q': 2, 'C': 3}
df["Embarked"] = df["Embarked"].map(mapping1)

mapping2 = {'male': 0, 'female': 1}
df["Sex"] = df["Sex"].map(mapping2)


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

#filling null values of age
df["Age"].fillna(df["Age"].mean(), inplace= True)

#since fare has one missing value, replacing it with the median
df["Fare"].fillna(df["Fare"].median(), inplace= True)


#scaling the fare column 
fare = np.array(df['Fare'])
df["Fare"] = MinMaxScaler(feature_range= (1,50)).fit_transform(fare.reshape(-1,1))

iqr = df["Fare"].quantile(0.75) - df["Fare"].quantile(0.25)
lower_bound = df["Fare"].quantile(0.25) - (iqr * 1.5)
upper_bound = df["Fare"].quantile(0.75) + (iqr * 1.5)

mean_fare = df[(df["Fare"] >= lower_bound) & (df["Fare"] <= upper_bound)]["Fare"].mean()
df.loc[df["Fare"] < lower_bound, "Fare"] = mean_fare
df.loc[df["Fare"] > upper_bound, "Fare"] = mean_fare

sns.boxplot(df["Fare"]) 
plt.tight_layout()
plt.title("Histogram of Fare after scaling and removing outliers")
plt.show()
plt.figure()

y = df["Survived"]
X = df.drop("Survived" , axis= 1)

X_train , X_test , Y_train  , Y_test = train_test_split(X,y, train_size = 0.8)

X_train = PCA(n_components=4).fit_transform(X_train)
X_test = PCA(n_components=4).fit_transform(X_test)

params = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
}

gscv = GridSearchCV(LogisticRegression() , param_grid=params)
model = gscv.fit(X_train, Y_train)

best_model = model.best_estimator_

best_model.fit(X_train , Y_train)
y_pred = best_model.predict(X_test)

sns.heatmap(confusion_matrix(Y_test , y_pred) , annot = True , xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1']) 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Accuracy of model  :" , accuracy_score(Y_test , y_pred))
precision, recall, fscore, _ = precision_recall_fscore_support(Y_test, y_pred)
print("precision of model  :" , precision[0])
print("recall of model  :" , recall[0])
print("fscore of model  :" , fscore[0])



