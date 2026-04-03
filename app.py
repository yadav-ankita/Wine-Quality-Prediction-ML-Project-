from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = pickle.load(open('wine_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        data = [
                     float(request.form['fixed_acidity']),
                     float(request.form['volatile_acidity']),
                     float(request.form['citric_acid']),
                     float(request.form['residual_sugar']),
                     float(request.form['chlorides']),
                     float(request.form['free_sulfur_dioxide']),
                     float(request.form['total_sulfur_dioxide']),
                     float(request.form['density']),
                     float(request.form['pH']),
                     float(request.form['sulphates']),
                    float(request.form['alcohol'])
                    ]
        
        # Convert into numpy array
        final_input = np.array(data).reshape(1, -1)

        # Apply scaling
        final_input = scaler.transform(final_input)

        # Predict
        prediction = model.predict(final_input)

        # Output
        #result = "Good Quality Wine 🍷" if prediction[0] == 1 else "Bad Quality Wine ❌"
        result=prediction[0]
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return str(e)

import os
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))