from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model + encoders
model = pickle.load(open("crop_yield_model.pkl", "rb"))
state_encoder = pickle.load(open("state_encoder.pkl", "rb"))
crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form['state']
    crop = request.form['crop']
    year = int(request.form['year'])
    rainfall = float(request.form['rainfall'])
    fertilizer = float(request.form['fertilizer'])
    pesticide = float(request.form['pesticide'])

    try:
        state_encoded = state_encoder.transform([state])[0]
        crop_encoded = crop_encoder.transform([crop])[0]
    except:
        return render_template('result.html', prediction="Invalid State or Crop", crop=None)

    X = [[state_encoded, crop_encoded, year, rainfall, fertilizer, pesticide]]
    prediction = model.predict(X)[0]

    return render_template('result.html', prediction=round(prediction, 2), crop=crop)

if __name__ == "__main__":
    app.run(debug=True)
