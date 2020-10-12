import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import catboost 


file = open('project.joblib','rb')
model = joblib.load(file)


st.title('Financial Inclusion Prediction')
st.subheader('Which Individual are most likely to have or use a bank account?')
st.write('Financial Inclusion remains one of the main obstacles to economic and human development in Africa. Traditionally, access to bank accounts has been regarded as an indicator of financial inclusion. It enables households to save and facilitate payments while also helping businesses build up their credit-worthiness and improve their access to other finance services. Therefore, access to bank accounts is an essential contributor to long-term economic growth.')

html_temp = """
    <div style ='background-color: wheat; padding:10px; border-radius:10px; border:10px; margin:10px'>
    <h2> Streamlit ML Web App </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.write('Please provide the following details for individuals')
country = st.sidebar.selectbox('Country - 0 for Kenya, 1 for Rwanda, 2 for Tanzania, 3 for Uganda',(0,1,2,3))
year = st.sidebar.selectbox('Year',('2016','2017','2018'))
gender_of_respondent = st.sidebar.selectbox('Gender_of_respondent - 0 for Female, 1 for Male',(0,1))
cellphone_access = st.sidebar.selectbox('Cellphone_access - 0 for No, 1 for Yes',(0,1))
location_type  = st.sidebar.selectbox('Location_type - 0 for Rural, 1 for Urban',(0,1))
age_of_respondent = st.sidebar.slider('Age_of_respondent',0,120,30)
education_level = st.sidebar.selectbox('Education_level - 0 for primary, 1 for Secondary education, 2 for Tertiary education, 3 for Vocational/Specialised training, 4 for No formal education , 5 for Other/Dont know/RTA',(0,1,2,3,4,5))
marital_status = st.sidebar.selectbox('Marital_status - 0 for Married/Living together, 1 for Single/Never Married, 2 for Widowed, 3 for Divorced/Seperated, 4 for Dont know',(0,1,2,3,4))
job_type = st.sidebar.selectbox('Job_type - 0 for Self employed, 1 for Informally employed, 2 for Farming and Fishing, 3 for Remittance Dependent, 4 for Formally employed Private, 5 for Other Income, 6 for No Income, 7 for Formally employed Government, 8 for Government Dependent, 9 for Dont Know/Refuse to answer',(0,1,2,3,4,5,6,7,8,9))






features = {'country ':country,
'gender_of_respondent':gender_of_respondent,
'year':year,
'cellphone_access':cellphone_access,
'location_type':location_type,
'age_of_respondent':age_of_respondent,
'education_level':education_level,
'marital_status':marital_status,
'job_type':job_type
}


if st.sidebar.button('Submit'):
    data = pd.DataFrame(features,index=[0,1])
    st.write(data)

    prediction = model.predict(data)
    proba = model.predict_proba(data)[1]

   

    if prediction[0] == 0:
        st.error('Individual do not have or use a bank account')
    else:
        st.success('Individual have or use a bank account')

    proba_df = pd.DataFrame(proba,columns=['Probability'],index=['None Financial Inclusive','Financial Inclusive'])
    proba_df.plot(kind='barh')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()




