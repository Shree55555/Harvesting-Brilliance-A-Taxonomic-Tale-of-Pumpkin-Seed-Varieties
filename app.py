import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Initialize the Flask app
app = Flask(__name__)

# Load the trained Random Forest model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    # Route to display the home page
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        # Retrieve values from the UI
        Area = float(request.form["Area"])
        Perimeter = float(request.form["Perimeter"])
        Major_Axis_Length = float(request.form['Major_Axis_Length'])
        Solidity = float(request.form['Solidity'])
        Extent = float(request.form['Extent'])
        Roundness = float(request.form['Roundness'])
        Aspect_Ration = float(request.form['Aspect_Ration'])
        Compactness = float(request.form['Compactness'])

        # Reshape to 2D array
        features_values = np.array([[Area, Perimeter, Major_Axis_Length, Solidity, Extent, Roundness, Aspect_Ration, Compactness]])
        
        # Convert to DataFrame
        df = pd.DataFrame(features_values, columns=['Area', 'Perimeter', 'Major_Axis_Length', 'Solidity', 'Extent', 'Roundness', 'Aspect_Ration', 'Compactness'])
        
        # Make prediction
        prediction = model.predict(df)
        
        if prediction[0] == 0:
            result = "Your seed lies in Çerçevelik class"
        elif prediction[0] == 1:
            result = "Your seed lies in Ürgüp Sivrisi class"
            
        text = "Hence, based on calculation: "
        return render_template("predict.html", prediction_text=text + str(result))
        
    # If GET request, just render the form
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=False, port=5000)