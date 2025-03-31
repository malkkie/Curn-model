# Interactive Random Forest Churn Prediction Model using Streamlit (Enhanced User-Friendliness)

# Importing libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('Churn_Modelling.csv')

# Drop irrelevant columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encoding categorical variables
label_encoder_geo = LabelEncoder()
data['Geography'] = label_encoder_geo.fit_transform(data['Geography'])
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

# Splitting data into features and target variable
X = data.drop('Exited', axis=1)
y = data['Exited']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
def user_input_features():
    st.sidebar.header("Provide Customer Information")
    st.sidebar.write("Enter the customer's details below to predict if they will stay with or leave the bank.")

    CreditScore = st.sidebar.slider('Credit Score (300=Poor, 850=Excellent)', 300, 850, 650)
    Geography = st.sidebar.selectbox('Country', label_encoder_geo.classes_)
    Gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
    Age = st.sidebar.slider('Age', 18, 100, 30)
    Tenure = st.sidebar.slider('Years with Bank (Tenure)', 0, 10, 5)
    Balance = st.sidebar.number_input('Current Account Balance (€)', min_value=0.0, max_value=250000.0, value=50000.0)
    NumOfProducts = st.sidebar.slider('Number of Bank Products Used', 1, 4, 1)
    HasCrCard = st.sidebar.selectbox('Does Customer Have a Credit Card?', ('No', 'Yes'))
    IsActiveMember = st.sidebar.selectbox('Is Customer Actively Engaged?', ('No', 'Yes'))
    EstimatedSalary = st.sidebar.number_input('Customer Estimated Salary (€)', min_value=0.0, max_value=500000.0, value=50000.0)

    data = {
        'CreditScore': CreditScore,
        'Geography': label_encoder_geo.transform([Geography])[0],
        'Gender': label_encoder_gender.transform([Gender])[0],
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': 1 if HasCrCard == 'Yes' else 0,
        'IsActiveMember': 1 if IsActiveMember == 'Yes' else 0,
        'EstimatedSalary': EstimatedSalary
    }

    features = pd.DataFrame(data, index=[0])
    return features

st.title('Customer Churn Prediction App')
st.write("This application predicts customer churn based on the provided information.")

input_df = user_input_features()
st.subheader('Provided Customer Details')
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction Outcome')
churn_labels = ['✅ Customer will likely stay', '⚠️ Customer is likely to churn']
st.write(churn_labels[prediction[0]])

st.subheader('Prediction Confidence')
st.write(prediction_proba)
