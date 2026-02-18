# Data manipulation
import numpy as np
import pandas as pd

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression

# NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def model_pipeline(df: pd.DataFrame, target_col: str, problem: str = "regression", num_classes: int = 2):
    # Preprocess data
    df = df.fillna(0)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    m = "linear regression" if problem == "regression" else "neural network"

    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    if problem == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build models
    if m == "linear regression":
        model = LinearRegression()
    elif m == "neural network":
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        raise ValueError("Unsupported model type")
    
    # Train model
    if m == "neural network":
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    elif m == "linear regression":
        model.fit(X_train, y_train)

   # Predictions
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1) if problem == "classification" else y_pred

    # Metrics
    if problem == "classification":
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    else:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R^2 Score: {r2:.3f}")

    # Save model
    if m == "linear regression":
        import joblib
        # Save model
        joblib.dump(model, "model.pkl")
        # Load model
        loaded_model = joblib.load("model.pkl")
    elif m == "neural network":
        model.save("model.h5")
        from tensorflow.keras.models import load_model
        loaded_model = load_model("model.h5")

    return return_model_stats(model, X_train, y_train, X_test, y_test, problem_type=problem, history=history if m == "neural network" else None)
    
def return_model_stats(model, X_train, y_train, X_test, y_test, problem_type="regression", history=None):
    stats = {}

    if problem_type == "regression":
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        stats['train_mse'] = mean_squared_error(y_train, y_pred_train)
        stats['train_r2'] = r2_score(y_train, y_pred_train)
        stats['test_mse'] = mean_squared_error(y_test, y_pred_test)
        stats['test_r2'] = r2_score(y_test, y_pred_test)
        stats['y_test'] = y_test
        stats['y_pred_test'] = y_pred_test

    else:  # classification
        y_pred_train = np.argmax(model.predict(X_train), axis=1)
        y_pred_test = np.argmax(model.predict(X_test), axis=1)

        stats['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        stats['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        stats['classification_report'] = classification_report(y_test, y_pred_test, output_dict=True)

        # Include loss/accuracy history if neural network
        if history:
            stats['history'] = history.history

    return stats