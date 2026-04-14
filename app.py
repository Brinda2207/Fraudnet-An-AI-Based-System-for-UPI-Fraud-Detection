from flask import Flask, render_template, request
import joblib
import numpy as np
import csv
import os
from datetime import datetime
import tensorflow as tf  # Added for Autoencoder

# Load Supervised Artifacts
model             = joblib.load('final_fraud_model.pkl')
scaler            = joblib.load('scaler.pkl')
features          = joblib.load('feature_list.pkl')
optimal_threshold = joblib.load('optimal_threshold.pkl')

# Load Unsupervised/Hybrid Artifacts
autoencoder       = tf.keras.models.load_model('autoencoder_model.keras')
hybrid_threshold  = joblib.load('hybrid_threshold.pkl')

DYNAMIC_DATA_PATH = 'dynamic_transactions.csv'
app = Flask(__name__, template_folder='templates')

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Capture Form Data [cite: 1]
        amount        = float(request.form.get('Transaction_Amount', 1000))
        hour_cat      = request.form.get('hour_category', 'morning')
        device_cat    = request.form.get('device_category', 'mobile')
        txn_type_cat  = request.form.get('txn_type_category', 'payment')
        location      = int(request.form.get('location_category', 0))
        payment_cat   = request.form.get('payment_category', 'upi')
        device_change = int(request.form.get('device_change', 0))
        txn_freq      = request.form.get('txn_frequency', 'low')
        prev_fraud    = int(request.form.get('prev_fraud', 0))
        account_age   = int(request.form.get('account_age', 365))

        # 2. Map Categorical Inputs [cite: 1]
        hour_map     = {'morning': 9, 'afternoon': 14, 'evening': 19, 'night': 2}
        device_map   = {'mobile': 0, 'desktop': 1, 'tablet': 2}
        txn_type_map = {'transfer': 0, 'payment': 1, 'withdrawal': 2}
        payment_map  = {'upi': 0, 'netbanking': 1, 'card': 2}
        freq_map     = {'low': 2, 'medium': 6, 'high': 12, 'very_high': 20}

        hour        = hour_map.get(hour_cat, 9)
        night_txn   = 1 if hour >= 22 or hour < 6 else 0
        device_used = device_map.get(device_cat, 0)
        txn_type    = txn_type_map.get(txn_type_cat, 1)
        payment     = payment_map.get(payment_cat, 0)
        num_txn_24h = freq_map.get(txn_freq, 2)

        feature_values = {
            'User_ID':                          5000,
            'Transaction_Amount':               amount,
            'Transaction_Type':                 txn_type,
            'Device_Used':                      device_used,
            'Location':                         location,
            'Previous_Fraudulent_Transactions': prev_fraud,
            'Account_Age':                      account_age,
            'Number_of_Transactions_Last_24H':  num_txn_24h,
            'Payment_Method':                   payment,
            'Hour':                             hour,
            'night_txn':                        night_txn,
            'device_change':                    device_change,
        }

        # 3. Preprocessing
        input_array  = np.array([feature_values[f] for f in features]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # 4. Hybrid Prediction Logic
        # Get Supervised Probability (XGBoost)
        xgb_prob = model.predict_proba(input_scaled)[0][1]
        
        # Get Unsupervised Anomaly Score (Autoencoder MSE)
        reconstructed = autoencoder.predict(input_scaled, verbose=0)
        mse = np.mean(np.power(input_scaled - reconstructed, 2), axis=1)[0]
        
        # Ensemble: Hybrid Score calculation
        # In your notebook, the hybrid score was (xgb_prob + mse) / 2
        hybrid_score = (xgb_prob + mse) / 2
        
        # Final Decision based on Hybrid Threshold
        prediction  = 1 if hybrid_score >= hybrid_threshold else 0
        result      = "⚠️ FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE TRANSACTION"

        # 5. Logging and Response [cite: 1]
        row = dict(feature_values)
        row['Fraudulent']        = int(prediction)
        row['Fraud_Probability'] = round(float(hybrid_score), 4)
        row['Timestamp']         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(DYNAMIC_DATA_PATH, 'a', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=row.keys())
            if not os.path.isfile(DYNAMIC_DATA_PATH):
                writer.writeheader()
            writer.writerow(row)

        return render_template("home.html",
                               result=result,
                               probability=round(float(hybrid_score) * 100, 2))
    except Exception as e:
        return render_template("home.html", error=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)