import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Load model and encoders safely
try:
    model = tf.keras.models.load_model('model.h5')

    with open('geo_encoder.pkl', 'rb') as file:
        geo_encoder = pickle.load(file)

    with open('ohe.pkl', 'rb') as file:
        ohe = pickle.load(file)

    with open('lable.pkl', 'rb') as file:
        label = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit UI
st.title("Customer Churn Prediction")

# Store session state to avoid reloading issues
if 'CreditScore' not in st.session_state:
    st.session_state['CreditScore'] = 0.0

CreditScore = st.number_input('Credit Score', min_value=0, key='CreditScore')
Geography = st.selectbox('Geography', ohe.categories_[0])
Gender = st.selectbox('Gender', label.classes_)
Age = st.slider('Age', 18, 92)
Tenure = st.slider('Tenure', 0, 10)
Balance = st.number_input('Balance', min_value=0.0)
NumOfProducts = st.slider('Number Of Products', 1, 4)
HasCrCard = st.selectbox('Has Credit Card', [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0)

# Ensure encoders are loaded correctly
if ohe is None or label is None or scaler is None:
    st.error("Error: Encoders or scaler are not loaded properly.")
    st.stop()

# Prepare input data
try:
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Gender': [label.transform([Gender])[0]],  # Encoding Gender
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    # One-hot encode Geography safely
    geo_encoded = ohe.transform([[Geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

    # Combine data
    final_input = pd.concat([geo_encoded_df, input_data], axis=1)

    # Ensure the column order is correct
    expected_columns = scaler.feature_names_in_

    # Handle missing columns
    missing_cols = set(expected_columns) - set(final_input.columns)
    for col in missing_cols:
        final_input[col] = 0

    final_input = final_input[expected_columns]  # Reorder

    # Check input before scaling
    if final_input.isnull().values.any():
        st.error("Error: Input contains NaN values.")
        st.stop()

    # Transform with scaler
    df_scaled = scaler.transform(final_input)

    # Make prediction
    prediction = model.predict(df_scaled)
    prediction_proba = prediction[0][0]

    # Display result
    if prediction_proba > 0.5:
        st.success(prediction_proba)
    else:
        st.success(prediction_proba)

except Exception as e:
    st.error(f"Processing error: {e}")
