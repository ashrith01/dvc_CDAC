import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

df= pd.read_csv('C:\\Users\\Lenovo\\Jupyter Notebook\\CDAC\\crop_recommendation.csv')

target_variable = None
for column in df.columns:
    if df[column].dtype == 'object':
        if len(df[column].unique()) <= 4:
            target_variable = column
            break

if target_variable is None:
    corr = df.corr()
    target_variable = corr.nlargest(1, 'target_variable')['target_variable'].index[0]

y = df[target_variable].values
X = df.drop(target_variable, axis=1).values
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()

null_values = df.isnull().sum()

# Print the number of null values in each column
print(null_values)

# Replace null values with column mean values
mean_values = df.mean()
df.fillna(mean_values, inplace=True)

def handle_outliers_IQR(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    return df

# Apply the handle_outliers_IQR function to each numeric column in the dataframe
for col in df.select_dtypes(include='number'):
    df = handle_outliers_IQR(df, col)






