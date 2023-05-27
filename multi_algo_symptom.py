import streamlit as st
import pickle

# Load the Decision Tree model
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

# Load the Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

# Load the KNN model
with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

# Load the Naive Bayes model
with open('naive_bayes_model.pkl', 'rb') as file:
    naive_bayes_model = pickle.load(file)

# Set the page title
st.title("Disease Prediction System")

# Function to predict disease using the four models
def predict_disease(symptoms):
    # Prepare the input data
    l2 = [0] * len(l1)
    for k in range(len(l1)):
        for z in symptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]

    # Use the Decision Tree model for prediction
    decision_tree_prediction = decision_tree_model.predict(inputtest)

    # Use the Random Forest model for prediction
    random_forest_prediction = random_forest_model.predict(inputtest)

    # Use the k-Nearest Neighbors model for prediction
    knn_prediction = knn_model.predict(inputtest)

    # Use the Naive Bayes model for prediction
    naive_bayes_prediction = naive_bayes_model.predict(inputtest)

    # Display the predictions
    st.subheader("Prediction Results:")
    st.write("Decision Tree Prediction:", decision_tree_prediction[0])
    st.write("Random Forest Prediction:", random_forest_prediction[0])
    st.write("k-Nearest Neighbors Prediction:", knn_prediction[0])
    st.write("Naive Bayes Prediction:", naive_bayes_prediction[0])

# Create input fields for symptoms
st.subheader("Enter Symptoms:")
symptom1 = st.text_input("Symptom 1")
symptom2 = st.text_input("Symptom 2")
symptom3 = st.text_input("Symptom 3")
symptom4 = st.text_input("Symptom 4")
symptom5 = st.text_input("Symptom 5")

symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]

# Check if all symptoms are entered
if all(symptoms):
    # Predict disease when the "Predict" button is clicked
    if st.button("Predict"):
        predict_disease(symptoms)
else:
    st.warning("Please enter all symptoms.")

