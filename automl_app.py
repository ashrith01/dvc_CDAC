import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

st.title('AutoML Web App')
st.subheader('Upload a Dataset')
st.subheader('Preprocessing and finding target variable will be done')
st.subheader('Classified using Four ML models : Logistic regression, Random Forest, KNN, SVM')
st.subheader('Results are shown')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file",type='csv')
if not uploaded_file:
        st.sidebar.write('Please upload an file before preceeding!')
        st.stop()
else:
    df = pd.read_csv(uploaded_file)

st.write('Sample of the data uploaded:')
st.dataframe(df.head()) 
st.write('These are the columns present in the data',df.columns)



target_variable = None
for column in df.columns[::-1]:
    if df[column].dtype == 'object':
        if len(df[column].unique()) <= 4:
            target_variable = column
            break

if target_variable is None:
    target_variable = df.columns[-1]

st.write('The Target Variable is: ',target_variable)
# create a LabelEncoder object
le = LabelEncoder()

# loop through each column in your dataset
for col in df.columns:
    # check if the column is categorical
    if df[col].dtype == 'object' and col!=target_variable:
        # use LabelEncoder to convert the categorical values to numeric
        df[col] = le.fit_transform(df[col])


null_values = df.isnull().sum()

# Print the number of null values in each column
#print(null_values)

# Replace null values with column mean values
mean_values = df.mean()
df.fillna(mean_values, inplace=True)


y = df[target_variable]
X = df.drop(target_variable, axis=1)
model = ExtraTreesClassifier()
model.fit(X,y)
 #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances_sorted = feat_importances.sort_values(ascending=False)
st.subheader('These are importance of features present in the data')
st.write(feat_importances_sorted)
st.bar_chart(feat_importances_sorted)

# Select the top 4 most important features
top_4_features = feat_importances_sorted[:4].index.tolist()

# Create a new DataFrame with only the top 4 features and the target variable
df= df[top_4_features + [target_variable]]

st.dataframe(df.head())


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


from sklearn.compose import ColumnTransformer
cat_cols = [col for col in df.columns if df[col].dtype == 'object' and col!=target_variable]
#print(cat_cols)


df = pd.get_dummies(df, columns = cat_cols)
#print(df)

# Print transformed data
#print(df.head())

# Standardize numerical variables using z-score normalization
num_cols = df.select_dtypes(include='number').columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


pred = df[target_variable]
x_content = df.drop(target_variable, axis=1)
from sklearn import preprocessing
from sklearn import utils
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(pred)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_content,y_transformed, test_size=0.3,random_state =2)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Fit random forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Fit KNN model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Fit SVM model
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

option = st.sidebar.selectbox(
    'Choose a model to test?',
    ('Logistic Regression', 'Random Forest','KNN','SVM'))

Genrate_pred = st.sidebar.button("Test")
if(Genrate_pred):
    if(option=='Logistic Regression'):
        st.write('Logistic Regression accuracy:', lr_accuracy) 

    elif(option=='Random Forest'):
        st.write('Random Forest accuracy:', rf_accuracy)

    elif(option=='KNN'):
        st.write('KNN accuracy:', knn_accuracy)

    elif(option=='SVM'):
        st.write('SVM accuracy:', svm_accuracy)


    #Find the best model
    models = {'Logistic Regression': lr_accuracy, 'Random Forest': rf_accuracy, 'KNN': knn_accuracy, 'SVM': svm_accuracy}
    best_model = max(models, key=models.get)

    # Print the best model and its accuracy
    st.write('Best model:', best_model)
    st.write('Best Model Accuracy:', models[best_model])

    results = pd.DataFrame(models,index=['Accuracy'])
    st.table(results)
   

    # Print classification report and confusion matrix for the best model
    if best_model == 'Logistic Regression':
        pred = lr_pred
    elif best_model == 'Random Forest':
        pred = rf_pred
    elif best_model == 'KNN':
        pred = knn_pred
    else:
        pred = svm_pred
        
    st.write('Classification Report:\n')
    st.text(classification_report(y_test, pred))
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(9,9))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap = 'Blues', ax = ax)    
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted Values")
    class_names = df[target_variable].unique()
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    st.pyplot(fig)








