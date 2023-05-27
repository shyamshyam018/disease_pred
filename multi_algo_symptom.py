# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn import tree, ensemble, neighbors, naive_bayes
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read the training and testing datasets
df_train = pd.read_csv("training.csv")
df_test = pd.read_csv("testing.csv")

# Extract the features and target variable from the datasets
X_train = df_train[l1]
y_train = df_train["prognosis"]

X_test = df_test[l1]
y_test = df_test["prognosis"]

# Train the models
clf1 = tree.DecisionTreeClassifier()
clf1.fit(X_train, y_train)

clf2 = ensemble.RandomForestClassifier(n_estimators=100)
clf2.fit(X_train, y_train)

clf3 = neighbors.KNeighborsClassifier()
clf3.fit(X_train, y_train)

clf4 = naive_bayes.GaussianNB()
clf4.fit(X_train, y_train)

# Define the prediction functions for each model
def decision_tree_prediction(symptoms):
    l2 = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1
    input_test = [l2]
    return clf1.predict(input_test)[0]

def random_forest_prediction(symptoms):
    l2 = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1
    input_test = [l2]
    return clf2.predict(input_test)[0]

def knn_prediction(symptoms):
    l2 = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1
    input_test = [l2]
    return clf3.predict(input_test)[0]

def naive_bayes_prediction(symptoms):
    l2 = [0] * len(l1)
    for symptom in symptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1
    input_test = [l2]
    return clf4.predict(input_test)[0]

# Define the Streamlit app
def main():
    st.title("Medical Diagnosis App")
    st.sidebar.title("Symptoms")

    # Collect user inputs for symptoms
    symptoms = []
    for i in range(5):
        symptom = st.sidebar.selectbox(f"Symptom {i+1}", options=l1, index=0)
        symptoms.append(symptom)

    # Display the selected symptoms
    st.sidebar.markdown("### Selected Symptoms")
    for symptom in symptoms:
        st.sidebar.write(symptom)

    # Run the models and display the predictions
    st.markdown("### Prediction Results")

    with st.beta_expander("Decision Tree"):
        if len(symptoms) >= 2:
            prediction = decision_tree_prediction(symptoms)
            st.write("Prediction:", disease[prediction])
        else:
            st.write("Kindly select at least two symptoms.")

    with st.beta_expander("Random Forest"):
        if len(symptoms) >= 2:
            prediction = random_forest_prediction(symptoms)
            st.write("Prediction:", disease[prediction])
        else:
            st.write("Kindly select at least two symptoms.")

    with st.beta_expander("K-Nearest Neighbors"):
        if len(symptoms) >= 2:
            prediction = knn_prediction(symptoms)
            st.write("Prediction:", disease[prediction])
        else:
            st.write("Kindly select at least two symptoms.")

    with st.beta_expander("Naive Bayes"):
        if len(symptoms) >= 2:
            prediction = naive_bayes_prediction(symptoms)
            st.write("Prediction:", disease[prediction])
        else:
            st.write("Kindly select at least two symptoms.")

if __name__ == "__main__":
    main()
