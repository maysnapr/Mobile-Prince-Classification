from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model_terbaik.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html', inputs=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inputs = request.form.to_dict()
        
        # Menyusun 20 fitur sesuai urutan dataset train.csv
        features = [
            float(inputs['battery_power']),
            int(inputs['blue']),
            float(inputs['clock_speed']),
            int(inputs['dual_sim']),
            float(inputs['fc']),
            int(inputs['four_g']),
            float(inputs['int_memory']),
            float(inputs['m_dep']),
            float(inputs['mobile_wt']),
            int(inputs['n_cores']),
            float(inputs['pc']),
            float(inputs['px_height']),
            float(inputs['px_width']),
            float(inputs['ram']),
            float(inputs['sc_h']),
            float(inputs['sc_w']),
            float(inputs['talk_time']),
            int(inputs['three_g']),
            int(inputs['touch_screen']),
            int(inputs['wifi'])
        ]
        
        # Standarisasi dan Prediksi
        std_features = scaler.transform([features])
        prediction = model.predict(std_features)[0]
        
        # Mapping label
        price_labels = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
        result = price_labels[prediction]
        
        return render_template('index.html', prediction_text=result, inputs=inputs)

if __name__ == "__main__":
    app.run(debug=True)