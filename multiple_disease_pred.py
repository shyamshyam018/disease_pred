import streamlit as st
from streamlit_option_menu import option_menu


def main():
    st.title("Multiple Disease Prediction System")
    st.markdown("Welcome to the Disease Prediction System. Please click the button below to get started.")

    if st.button("Get Started"):
        st.sidebar.title("Multiple Disease Prediction System")
        st.sidebar.markdown("Select a disease for prediction:")

        selected = option_menu(
            "Select a Disease",
            ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
            icons=["activity", "heart", "person"],
            default_index=0,
        )

        if selected == "Diabetes Prediction":
            display_diabetes_prediction()
        elif selected == "Heart Disease Prediction":
            display_heart_disease_prediction()
        elif selected == "Parkinsons Prediction":
            display_parkinsons_prediction()


def display_diabetes_prediction():
    st.title("Diabetes Prediction using ML")
    # Rest of the code for Diabetes Prediction page...


def display_heart_disease_prediction():
    st.title("Heart Disease Prediction using ML")
    # Rest of the code for Heart Disease Prediction page...


def display_parkinsons_prediction():
    st.title("Parkinson's Disease Prediction using ML")
    # Rest of the code for Parkinson's Prediction page...


if __name__ == "__main__":
    main()
