from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("student_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    spi = float(request.form['spi'])
    test = float(request.form['test'])
    assembly = float(request.form['assembly'])
    
    features = np.array([[spi, test, assembly]])
    prediction = model.predict(features)[0]
    
    return render_template('index.html', 
                           prediction_text=f"Predicted Final Marks: {round(prediction,2)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

