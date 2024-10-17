
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset (assuming it's in a CSV file)
def load_data():
    data = pd.read_csv('Iris.csv', header=None, skiprows=1)
    data = data[0].str.split(',', expand=True)
    data.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Variety']
    data['Variety'] = data['Variety'].str.replace('"', '')
    data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].astype(float)
    return data

# Load and prepare data
iris_data_fixed = load_data()

# Split features and labels
X = iris_data_fixed[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = iris_data_fixed['Variety']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Streamlit app
st.title('Flower Classification App')

st.write("""
### Enter the parameters of the flower to predict its class:
""")

# Input fields for flower parameters
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button('Predict'):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = clf.predict(input_data)
    predicted_class = label_encoder.inverse_transform(prediction)

    # Show result
    st.write(f'The predicted class is: **{predicted_class[0]}**')

