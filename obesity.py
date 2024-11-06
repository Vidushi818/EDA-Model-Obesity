# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('OneDrive/Desktop/ObesityDataSet.csv')

# %%
df

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.describe()

# %%
df.columns

# %%
df.nunique()

# %%
df['NObeyesdad'].unique()

# %%
df.isnull().sum()

# %%
student = df.drop(['MTRANS','CALC','CAEC','FAVC','TUE','FAF'],axis = 1)

# %%
student.head()

# %%
len(student.columns)

# %%
plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
sns.barplot(x = "NObeyesdad" , y = "Age",data = df);
plt.show()

# %%
gender_counts = df.Gender.value_counts()
gender_counts

# %%
plt.figure(figsize=(12,6))
plt.pie(gender_counts, labels = gender_counts.index, autopct="%1.2f%%")

# %%
corelation = student.select_dtypes(include=[float, int]).corr()

# %%
corelation

# %%
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)

# %%
sns.pairplot(student)

# %%
sns.relplot(x='Height', y='Weight', hue='Gender', data = student)

# %%
sns.histplot(student['CH2O'])

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# %%
obese_categories = ['Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
df['Obese'] = df['NObeyesdad'].apply(lambda x: 1 if x in obese_categories else 0)

# %%
X = df.drop(['NObeyesdad'],axis=1)
y= df['Obese']

# %%
categorical_features = ['Gender', 'family_history_with_overweight', 'SMOKE', 'SCC']
numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O']

# %%
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features),('cat', OneHotEncoder(), categorical_features)])

# %%
model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression(max_iter=1000))])

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# %%
X_train

# %%
X_test

# %%
X_train.shape

# %%
model.fit(X_train, y_train)

# %%
y_pred= model.predict(X_test)

# %%
print(classification_report(y_test, y_pred))

# %%



