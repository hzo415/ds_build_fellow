import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import r2_score, mean_squared_error # type: ignore

# Load the data
#st.write("Ignore Errors Below will be removed After file is inserted")
#uploaded_file = st.file_uploader("Upload the healthcare data file")
#df = pd.read_csv(uploaded_file)

df = pd.read_csv('https://raw.githubusercontent.com/hzo415/ds_build_fellow/refs/heads/main/HealthData.csv')


# Programmatically set the config to hide error details
st.set_option('client.showErrorDetails', False)

# Define features and target variable
X = df.drop('charges', axis=1)
y = df['charges']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'children']),
        ('cat', OneHotEncoder(drop='first'), ['sex', 'smoker', 'region'])
    ]
)
X = df.drop('charges', axis=1)
y = df['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing to features
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create and train model
model = LinearRegression()
model.fit(X_train_processed, y_train)

# Make predictions on test set
y_pred = model.predict(X_test_processed)

# Evaluate performance using R² and RMSE
r2 = r2_score(y_test, y_pred)


# Set title and introduction
st.title("Insurance Charges Predictor")
st.markdown("Welcome to the Insurance Charges Predictor!")
st.write("Enter the details below to predict insurance charges:")


# Input fields with error messages Input fields
max_age = 100
min_age = 18
age = st.slider("Age", min_value=min_age, max_value=max_age, value=30, help="Age should be between 18 and 100.")

sex = st.selectbox("Sex", options=['male','female'], help="Please select a valid sex.")

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, help="BMI should be between 10 and 60.")

children = st.number_input("Number of Children", min_value=0, max_value=5, value=0, help="Number of children should be between 0 and 5.")

smoker = st.selectbox("Smoker", options=['yes','no'], help="Please select a valid smoker status.")

region = st.selectbox("Region", options=['northeast','northwest', 'southeast', 'southwest'], help="Please select a valid region.")

# Button to predict charges
if st.button("Predict Charges"):
    try:
        # User inputs
        input_df = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children':[children],
            'smoker':[smoker],
            'region':[region]
        })

        # Transform data
        input_processed = preprocessor.transform(input_df)

        # Predict charges
        prediction = model.predict(input_processed)[0]
    

        # Display result
        st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")
        st.success(f"R² Score: {r2:.3f}")

        # Plot actual vs predicted charges
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='skyblue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line (y=x)
        ax.set_xlabel('Actual Charges')
        ax.set_ylabel('Predicted Charges')
        ax.set_title('Actual vs Predicted Charges')
        st.pyplot(fig)

    except Exception as e:
        st.error(str(e))

