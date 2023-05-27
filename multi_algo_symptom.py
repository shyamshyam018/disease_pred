import streamlit as st
import pandas as pd
from sklearn import tree, ensemble, neighbors, naive_bayes
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Define the list of symptoms
symptoms = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria',
    'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
    'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads',
    'scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails',
    'blister','red_sore_around_nose','yellow_crust_ooze']

diseases = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo']

# Read the training and testing datasets
df_train = pd.read_csv("training.csv")
df_test = pd.read_csv("testing.csv")

# Check for missing values in X_train and X_test
print(X_train.isna().sum())
print(X_test.isna().sum())

# Check for infinite values in X_train and X_test
print(np.isfinite(X_train).sum())
print(np.isfinite(X_test).sum())


# Extract the features and target variable from the datasets
X_train = df_train[symptoms]
y_train = df_train["prognosis"]

X_test = df_test[symptoms]
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
    input_test = [symptoms]
    return clf1.predict(input_test)[0]

def random_forest_prediction(symptoms):
    input_test = [symptoms]
    return clf2.predict(input_test)[0]

def knn_prediction(symptoms):
    input_test = [symptoms]
    return clf3.predict(input_test)[0]

def naive_bayes_prediction(symptoms):
    input_test = [symptoms]
    return clf4.predict(input_test)[0]

# Define the Streamlit app
def main():
    st.title("Medical Diagnosis App")
    st.sidebar.title("Symptoms")

    # Collect user inputs for symptoms
    symptoms = []
    for i in range(5):
        symptom = st.sidebar.selectbox(f"Symptom {i+1}", options=symptoms, index=0)
        symptoms.append(symptom)

    # Display the selected symptoms
    st.sidebar.markdown("### Selected Symptoms")
    for symptom in symptoms:
        st.sidebar.write(symptom)

    # Run the models and display the predictions
    st.markdown("### Prediction Results")

    with st.expander("Decision Tree"):
        if len(symptoms) >= 2:
            prediction = decision_tree_prediction(symptoms)
            st.write("Prediction:", diseases[prediction])
        else:
            st.write("Kindly select at least two symptoms.")

    with st.expander("Random Forest"):
        if len(symptoms) >= 2:
            prediction = random_forest_prediction(symptoms)
            st.write("Prediction:", diseases[prediction])
        else:
            st.write("Kindly select at least two symptoms.")

    with st.expander("K-Nearest Neighbors"):
        if len(symptoms) >= 2:
            prediction = knn_prediction(symptoms)
            st.write("Prediction:", diseases[prediction])
        else:
            st.write("Kindly select at least two symptoms.")

    with st.expander("Naive Bayes"):
        if len(symptoms) >= 2:
            prediction = naive_bayes_prediction(symptoms)
            st.write("Prediction:", diseases[prediction])
        else:
            st.write("Kindly select at least two symptoms.")

if __name__ == "__main__":
    main()
