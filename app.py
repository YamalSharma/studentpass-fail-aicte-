import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('student_model.pkl', 'rb'))

# Define a list of features used in training (from student-mat.csv)
feature_names = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
    'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
    'health', 'absences'
]

# Create a default input dictionary (fill with common or neutral values)
input_dict = {
    'school': 1,        # assumed encoded value
    'sex': 1,
    'age': 17,
    'address': 1,
    'famsize': 1,
    'Pstatus': 1,
    'Medu': 2,
    'Fedu': 2,
    'Mjob': 1,
    'Fjob': 1,
    'reason': 1,
    'guardian': 1,
    'traveltime': 1,
    'studytime': 2,     # user input
    'failures': 0,      # user input
    'schoolsup': 0,     # user input
    'famsup': 1,        # user input
    'paid': 0,
    'activities': 1,
    'nursery': 1,
    'higher': 1,
    'internet': 1,      # user input
    'romantic': 0,
    'famrel': 3,
    'freetime': 3,
    'goout': 3,
    'Dalc': 1,
    'Walc': 1,
    'health': 3,
    'absences': 2       # user input
}

# UI: Take values for only the key features
st.title("🎓 Student Performance Predictor")

studytime = st.selectbox("Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
absences = st.slider("Number of Absences", 0, 100, 5)
famsup = st.selectbox("Family Support", ["yes", "no"])
schoolsup = st.selectbox("School Support", ["yes", "no"])
internet = st.selectbox("Internet Access at Home", ["yes", "no"])
Medu = st.slider("Mother's Education (0 to 4)", 0, 4)
Fedu = st.slider("Father's Education (0 to 4)", 0, 4)

# Update inputs based on user
input_dict['studytime'] = studytime
input_dict['failures'] = failures
input_dict['absences'] = absences
input_dict['famsup'] = 1 if famsup == "yes" else 0
input_dict['schoolsup'] = 1 if schoolsup == "yes" else 0
input_dict['internet'] = 1 if internet == "yes" else 0
input_dict['Medu'] = Medu
input_dict['Fedu'] = Fedu

# Convert dict to dataframe row
input_df = pd.DataFrame([input_dict])[feature_names]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("✅ The student is likely to PASS.")
    else:
        st.error("❌ The student is likely to FAIL.")
