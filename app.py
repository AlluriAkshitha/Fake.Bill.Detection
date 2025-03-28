from flask import Flask, render_template, request
import numpy as np
import joblib
import gzip

app = Flask(__name__)

# Load the compressed ML model
with gzip.open("fake_bill_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data safely
        diagonal = float(request.form.get("diagonal", 0))
        length = float(request.form.get("length", 0))
        height_left = float(request.form.get("height_left", 0))
        height_right = float(request.form.get("height_right", 0))
        margin_up = float(request.form.get("margin_up", 0))
        margin_low = float(request.form.get("margin_low", 0))  # Ensure margin_low exists

        # Create a NumPy array for model prediction
        features = np.array([[diagonal, length, height_left, height_right, margin_up, margin_low]])
        prediction = model.predict(features)

        # Convert prediction to a readable format
        result = "Genuine" if prediction[0] == 1 else "Fake"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return f"Error: {e}", 400  # Return error message with a bad request response

if __name__ == "__main__":
    app.run(debug=True)
