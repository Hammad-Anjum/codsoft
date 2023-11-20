import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score , precision_recall_fscore_support
pd.set_option('display.max_columns', None)

df = pd.read_csv('creditcard.csv')

print(df.info())
print(df.describe())
print("missing values count : " , df.isna().sum())

#Missing values 
missingno.matrix(df)
plt.title("missing values")
plt.show()
plt.figure()

#count of each type of transaction
df['Class'].map({0: 'legit', 1: 'fraud'}).value_counts().plot(kind='pie' , autopct='%1.1f%%' , pctdistance=0.85, labeldistance=1.1)
plt.title("Count of transaction type")
plt.show()
plt.figure()

#for col in df.columns:
#    if col != 'Time' or col != 'Amount' or col != 'Class':
#        sns.scatterplot(x = col, y = 'Amount' , data = df , hue = 'Class')
#        plt.show()
#
#         plt.figure()


print(df.corr())

#sns.pairplot(df)

x = df.drop(["Time" , "Class"] , axis = 1)
y = df["Class"]

X_train , X_test , Y_train  , Y_test = train_test_split(x,y, train_size = 0.8,)

model = LogisticRegression().fit(X_train, Y_train)

y_pred = model.predict(X_test)

sns.heatmap(confusion_matrix(Y_test , y_pred) , annot = True) 
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Accuracy of model  :" , accuracy_score(Y_test , y_pred))

precision, recall, fscore, _ = precision_recall_fscore_support(Y_test, y_pred, average='macro')

print("precision of model  :" , precision)
print("recall of model  :" , recall)
print("fscore of model  :" , fscore)
