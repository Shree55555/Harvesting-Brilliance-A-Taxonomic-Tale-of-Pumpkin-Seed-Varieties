import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# -------------------- ROUTES --------------------

# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# About Page (NEW)
@app.route('/about')
def about():
    return render_template('about.html')


# Predict Page
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get form values safely
            Area = float(request.form.get("Area"))
            Perimeter = float(request.form.get("Perimeter"))
            Major_Axis_Length = float(request.form.get("Major_Axis_Length"))
            Solidity = float(request.form.get("Solidity"))
            Extent = float(request.form.get("Extent"))
            Roundness = float(request.form.get("Roundness"))
            Aspect_Ration = float(request.form.get("Aspect_Ration"))
            Compactness = float(request.form.get("Compactness"))

            # Convert to array
            features_values = np.array([[Area, Perimeter, Major_Axis_Length,
                                         Solidity, Extent, Roundness,
                                         Aspect_Ration, Compactness]])

            # Convert to DataFrame (important for column alignment)
            df = pd.DataFrame(features_values, columns=[
                'Area', 'Perimeter', 'Major_Axis_Length',
                'Solidity', 'Extent', 'Roundness',
                'Aspect_Ration', 'Compactness'
            ])

            # Prediction
            prediction = model.predict(df)

            # Output mapping
            if prediction[0] == 0:
                result = "🌱 Your seed belongs to Çerçevelik variety"
            elif prediction[0] == 1:
                result = "🌿 Your seed belongs to Ürgüp Sivrisi variety"
            else:
                result = "⚠️ Unknown class detected"

            text = "Prediction Result: "
            return render_template("predict.html", prediction_text=text + result)

        except Exception as e:
            return render_template("predict.html",
                                   prediction_text=f"Error: {str(e)}")

    return render_template("predict.html")


# -------------------- RUN APP --------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)