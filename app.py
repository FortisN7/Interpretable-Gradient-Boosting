import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import lightgbm as lgbm
import shap
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# Define the objective function for Optuna to minimize
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
        early_stopping_rounds=100,
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
study.optimize(objective, n_trials=100)

# Train the model with the best hyperparameters
best_params = study.best_params
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)

# Generate the SHAP values and summary plot
explainer = shap.Explainer(model)
shap_values = explainer(test)

shap.summary_plot(shap_values, test, max_display=15, cmap='seismic')

# Generate the feature importance plot
lgbm.plot_importance(model, max_num_features=15)