from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load artifacts
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")
columns = joblib.load("models/columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    input_data = request.form.to_dict()

    # Convert to float
    input_data = {
        k: float(v) if v != "" else np.nan
        for k, v in input_data.items()
    }

    df = pd.DataFrame([input_data])

    # Ensure all required columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[columns]

    # Impute
    df_imputed = pd.DataFrame(
        imputer.transform(df),
        columns=columns
    )

    # Scale
    df_scaled = pd.DataFrame(
        scaler.transform(df_imputed),
        columns=columns
    )

    # Predict
    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0][1]

    result = "Fraud" if prediction == 1 else "Legit"
    probability = round(proba * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)
