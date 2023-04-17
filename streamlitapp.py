import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import lightgbm as lgbm
import numpy as np
import shap
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

shap.initjs()
st.set_option('deprecation.showPyplotGlobalUse', False)

# MODEL

training = st.title("Model is training...")

# Define the objective function for Optuna to minimize
@st.cache_data
def load_data():
    def objective(trial):
        params = {
            #         "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.95, step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.95, step=0.1
            ),
        }

        X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2)
        
        model = LGBMRegressor(**params)
        model.fit(
            X_train_, y_train_,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=200,
            verbose=False
        )

        y_pred = model.predict(X_val)
        MSE = mean_squared_error(y_val, y_pred, squared=False)
        return MSE

    # Load and preprocess the data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    X_train = train.iloc[:, 1:-1]
    y_train = train.iloc[:, -1]

    X_train = pd.get_dummies(X_train)
    test = pd.get_dummies(test)

    missing_cols = set(X_train.columns) - set(test.columns)
    for c in missing_cols:
        test[c] = 0
    test = test[X_train.columns]

    # Define the study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100) # Can make larger for better accuracy (not required)

    # Train the model with the best hyperparameters
    best_params = study.best_params
    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)

    # Generate the SHAP values (outputs summary plot when button is pressed)
    # explainer = shap.Explainer(model)
    # shap_values = explainer(test)

    # shap.summary_plot(shap_values, test, max_display=15, cmap='seismic')

    # explainer = shap.TreeExplainer(model)
    # shap_interaction = explainer.shap_interaction_values(test)

    # shap.summary_plot(shap_interaction, test)

    # Generate the feature importance plot
    # lgbm.plot_importance(model, max_num_features=15)

    return model, X_train, test
#-------------------------------------------------------------------------------------------------------------------------

training.empty()

model, X_train, test = load_data()
# APP

def app():
    st.markdown("<body style ='color:#E2E0D9;'></body>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; color: #1B9E91;'>House Price Prediction in Ames, Iowa</h4>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center; color: #1B9E91;'>A LightGBM model is used to estimate the range of house prices based on your selection. The modeling process is done using the data found on Kaggle (link at left bottom corner of page)</h5>", unsafe_allow_html=True)

    # Set the title
    st.title("House Price Prediction in Ames, Iowa by Nick Fortis")

    # Set up the left sidebar
    sliders = {}
    for col in X_train.columns:
        sliders[col] = st.sidebar.slider(col, float(X_train[col].min()), float(X_train[col].max()), float(X_train[col].min()))
    st.sidebar.markdown("[Kaggle Link to Data Set](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)")

    # Set up the main panel
    if st.button("Calculate Estimated House Price"):
        # Create a DataFrame with the input values
        input_df = pd.DataFrame.from_dict(sliders, orient='index').T
        input_df = pd.get_dummies(input_df)
        missing_cols = set(X_train.columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0
        input_df = input_df[X_train.columns]

        # Get the model's predicted value for the input
        prediction = model.predict(input_df)[0]

        # Display the predicted value
        st.subheader("Predicted House Price")
        st.write("\$", round(prediction*0.85,2), " - ", "\$", round(prediction * 1.15,2))

        # Display the Plots
        explainer = shap.Explainer(model)
        shap_values = explainer(test)

        st.subheader("SHAP Summary Plot")
        input = shap.summary_plot(shap_values, input_df, plot_type="bar", max_display=15, cmap='seismic')
        st.pyplot(input)

        explainer = shap.TreeExplainer(model)
        shap_interaction = explainer.shap_interaction_values(test)

        st.subheader("SHAP Interaction Plot")
        input = shap.summary_plot(shap_interaction, test)
        st.pyplot(input)

# Run the app
if __name__ == "__main__":
    app()