import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
#st.write('Hello world!')

st.title('Binary and Multi Classification Web App')
st.sidebar.title('Classification')
## load file
#st.sidebar.write("# File Required")
#uploaded_image = st.sidebar.file_uploader('', type=['jpg','png','jpeg'])
uploaded_file = st.sidebar.file_uploader("Choose a file")
dataframe = pd.read_csv(uploaded_file)
st.write(dataframe)
option = st.sidebar.selectbox(
     'Which classification do you want to perform?',
     ('Binary Class', 'Multi Class'))

map_class = {
        0:'apple', 1:'banana', 2:'blackgram', 3:'chickpea', 4:'coconut', 5:'coffee', 6:'cotton',
 7:'grapes', 8:'jute', 9:'kidneybeans', 10:'lentil', 11:'maize', 12:'mango', 13:'mothbeans',
 14:'mungbean', 15:'muskmelon', 16:'orange', 17:'papaya', 18:'pigeonpeas', 19:'pomegranate',
 20:'rice', 21:'watermelon'
        }

map_class_bi ={0:'Kecimen',1:'Bensi'}
        
#Dataframe 
dict_class = {
        'Most Suitable Crop': ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton',
 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
 'rice', 'watermelon'],
        'Prob': [0]*22
        }

dict_class_bi ={ 'Raisin':['Kecimen','Besni'],'Prob':[0]*2}

df_results_bi= pd.DataFrame(dict_class_bi, columns = ['Raisin', 'Prob'])
        
df_results_multi = pd.DataFrame(dict_class, columns = ['Most Suitable Crop', 'Prob'])

def predictions_binary(preds):
    for i in range(2):
        df_results_bi.loc[df_results_bi['Raisin'].index[i], 'Prob'] = preds[0][i]
        
    return (df_results_bi)

def predictions_multi(preds):
    for i in range(22):
        df_results_multi.loc[df_results_multi['Most Suitable Crop'].index[i], 'Prob'] = preds[0][i]
        
    return (df_results_multi)

if(option=='Binary Class'):
    st.write('Fill the attributes')
    mal = st.number_input('Enter MajorAxisLength',max_value=997.291941)
    mial = st.number_input('Enter MinorAxisLength',max_value=492.275279)
    ecc = st.number_input('Enter Eccentricity',max_value=0.962124)
    ext = st.number_input('Enter Extent',max_value=0.835455)
    att= np.array([mal,mial,ecc,ext])
    att=att.reshape(1, -1)
    Genrate_pred = st.button("Detect Result")
    if Genrate_pred:
        st.subheader('Probabilities by Class')
        loaded_model = pickle.load(open('MLPpickle_file', 'rb'))
        preds = loaded_model.predict_proba(att)
        lab = loaded_model.predict(att)
        preds
        lab
        st.dataframe(predictions_binary(preds))
        st.subheader("The Raisin type is {}".format(map_class_bi[lab[0]]))


if(option=='Multi Class'):
    st.write('Fill the attributes')
    N_content = st.number_input('Enter Nitrogen Content',max_value=140)
    P_content = st.number_input('Enter Phosphorous Content',max_value=145)
    K_content = st.number_input('Enter Pottasium Content',max_value=205)
    temp = st.number_input('Enter temperature value',max_value=43.675)
    hum = st.number_input('Enter humidity value',max_value=99.98)
    ph = st.number_input('Enter ph value',max_value=9.93)
    rain = st.number_input('Enter rainfall value',max_value=298.56)
    att= np.array([N_content,P_content,K_content,temp,hum,ph,rain])
    att=att.reshape(1, -1)
    Genrate_pred = st.button("Detect Result") 
    if Genrate_pred:
        st.subheader('Probabilities by Class')
        loaded_model = pickle.load(open('knnpickle_file', 'rb'))
        preds = loaded_model.predict_proba(att)
        lab = loaded_model.predict(att)
        st.dataframe(predictions_multi(preds))
        st.subheader("The Most Suitable Crop to grow is {}".format(lab[0]))










