from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load models safely
try:
    scaler = pickle.load(open('model/StandardScaler.pkl', 'rb'))
    model = pickle.load(open('model/Modelforprediction.pkl', 'rb'))
except Exception as e:
    print("Error loading model:", e)


# Home route
@app.route('/')
def home():
    return render_template('home.html')


# Prediction route
@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'POST':
        try:
            # Get form data
            data = [float(request.form.get(x)) for x in [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]]

            # Convert to numpy array
            data_array = np.array(data).reshape(1, -1)

            # Scale data
            scaled_data = scaler.transform(data_array)

            # Prediction
            prediction = model.predict(scaled_data)[0]

            # Probability (for chart)
            probability = model.predict_proba(scaled_data)[0][1] * 100

            # Result text
            result = "Diabetic" if prediction == 1 else "Non Diabetic"

            return render_template(
                'single_prediction.html',
                result=result,
                prob=round(probability, 2)
            )

        except Exception as e:
            return f"Error occurred: {e}"

    # GET request
    return render_template('home.html')


# Run app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)