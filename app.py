from flask import Flask, render_template, request
import joblib
import numpy as np
import h5py
import csv
import os
import io
import base64
from datetime import datetime

# ── matplotlib non-interactive backend (must be set before pyplot import) ────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd
import networkx as nx
import shap

# ── Load XGBoost models ───────────────────────────────────────────────────────
model             = joblib.load('models/final_fraud_model.pkl')
scaler            = joblib.load('models/scaler.pkl')
features          = joblib.load('models/feature_list.pkl')
optimal_threshold = joblib.load('models/optimal_threshold.pkl')
hybrid_threshold  = joblib.load('models/hybrid_threshold.pkl')
mse_stats         = joblib.load('models/mse_stats.pkl')
_mse_p50          = mse_stats['p50']
_mse_p95          = mse_stats['p95']

# ── Load Rajeshwari's Random Forest (used for SHAP explanations) ──────────────
# Rajeshwari must run this once in her notebook and share the file:
#   import joblib
#   joblib.dump(rf, drive_path + 'rf_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')

# ── Load transaction dataset for graph analysis ───────────────────────────────
df_full = pd.read_csv('dynamic_transactions.csv')

# ── Load Autoencoder weights via h5py (no TensorFlow needed at inference) ────
with h5py.File('models/autoencoder_weights.weights.h5', 'r') as f:
    _ae_w1 = f['layers/dense/vars/0'][:]
    _ae_b1 = f['layers/dense/vars/1'][:]
    _ae_w2 = f['layers/dense_1/vars/0'][:]
    _ae_b2 = f['layers/dense_1/vars/1'][:]


def autoencoder_predict(x):
    """Pure numpy forward pass — identical to Keras relu encoder + linear decoder."""
    hidden = np.maximum(0, x @ _ae_w1 + _ae_b1)
    return hidden @ _ae_w2 + _ae_b2


# ── Helper: matplotlib figure → base64 PNG string ────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120,
                facecolor='#0f172a')          # matches your dark UI background
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


# ── SHAP waterfall chart for a single transaction ─────────────────────────────
def generate_shap_chart(feature_values_dict):
    """
    feature_values_dict : the raw (unscaled) feature dict from the form.
    Returns              : base64 PNG string of a SHAP waterfall chart.
    """
    # Build a readable-label version of the input for display
    label_map = {
        'User_ID':                          'User ID',
        'Transaction_Amount':               'Amount (₹)',
        'Transaction_Type':                 'Txn Type',
        'Device_Used':                      'Device',
        'Location':                         'Location',
        'Previous_Fraudulent_Transactions': 'Prev Frauds',
        'Account_Age':                      'Account Age',
        'Number_of_Transactions_Last_24H':  'Txns Last 24H',
        'Payment_Method':                   'Payment Method',
        'Hour':                             'Hour of Day',
        'night_txn':                        'Night Txn?',
        'device_change':                    'Device Change?',
    }

    input_array = np.array([feature_values_dict[f] for f in features]).reshape(1, -1)

    # Scale using the shared scaler (RF was trained on scaled data too)
    input_scaled = scaler.transform(input_array)
    input_df = pd.DataFrame(input_scaled, columns=[label_map.get(f, f) for f in features])

    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_df)

    # shap_values shape: (1, 12, 2) — pick index 1 = fraud class
    explanation = shap.Explanation(
        values        = shap_values[0][:, 1],
        base_values   = explainer.expected_value[1],
        data          = input_df.iloc[0].values,
        feature_names = input_df.columns.tolist()
    )

    plt.style.use('dark_background')
    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor('#0f172a')
    fig.set_size_inches(8, 5)
    return fig_to_base64(fig)


# ── Graph neighbourhood chart for a given location ───────────────────────────
def generate_graph_chart(location_val, feature_values_dict):
    """
    location_val        : the encoded location integer (0 / 1 / 2)
    feature_values_dict : full feature dict so we can flag this user's profile
    Returns             : base64 PNG string of the NetworkX graph, or None.
    """
    # Filter users who share the same location
    df_loc = df_full[df_full['Location'] == location_val].copy()
    if df_loc.empty:
        return None

    # Cap at 80 users so the graph stays readable
    sample_users = df_loc['User_ID'].unique()[:80]
    df_loc = df_loc[df_loc['User_ID'].isin(sample_users)]

    # Build graph: connect users who share the same Location
    G = nx.Graph()
    for user in df_loc['User_ID'].unique():
        G.add_node(user)

    for loc in df_loc['Location'].unique():
        users_at_loc = df_loc[df_loc['Location'] == loc]['User_ID'].unique()
        for i in range(len(users_at_loc)):
            for j in range(i + 1, len(users_at_loc)):
                G.add_edge(users_at_loc[i], users_at_loc[j])

    fraud_set  = set(df_loc[df_loc['Fraudulent'] == 1]['User_ID'].unique())

    # Check mule-like accounts (high txn freq + prior fraud history)
    mule_set = set(df_loc[
        (df_loc['Number_of_Transactions_Last_24H'] > 10) &
        (df_loc['Previous_Fraudulent_Transactions'] > 0)
    ]['User_ID'].unique())

    # Colour coding
    colors, sizes = [], []
    for node in G.nodes():
        if node in mule_set:
            colors.append('#f39c12')   # orange = mule / high-risk
            sizes.append(300)
        elif node in fraud_set:
            colors.append('#ef4444')   # red = confirmed fraud
            sizes.append(150)
        else:
            colors.append('#3b82f6')   # blue = normal
            sizes.append(60)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax,
            node_color=colors, node_size=sizes,
            with_labels=False,
            edge_color='#334155', width=0.6, alpha=0.9)

    mule_p   = mpatches.Patch(color='#f39c12', label='High-risk / mule')
    fraud_p  = mpatches.Patch(color='#ef4444', label='Confirmed fraud')
    normal_p = mpatches.Patch(color='#3b82f6', label='Normal user')
    ax.legend(handles=[mule_p, fraud_p, normal_p],
              loc='upper left', fontsize=9,
              facecolor='#1e293b', labelcolor='white', edgecolor='#334155')

    loc_labels = {0: 'Local', 1: 'Domestic', 2: 'International'}
    ax.set_title(f'Transaction network — {loc_labels.get(location_val, "")} location',
                 fontsize=11, fontweight='bold', color='#e5e7eb')
    return fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────────────
DYNAMIC_DATA_PATH = 'dynamic_transactions.csv'
app = Flask(__name__, template_folder='templates')


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Capture Form Data
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

        # 2. Map Categorical Inputs
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

        # 4. Hybrid Prediction — XGBoost + Autoencoder
        xgb_prob      = model.predict_proba(input_scaled)[0][1]
        reconstructed = autoencoder_predict(input_scaled)
        mse           = np.mean(np.power(input_scaled - reconstructed, 2), axis=1)[0]
        mse_norm      = float(np.clip((mse - _mse_p50) / (_mse_p95 - _mse_p50), 0, 1))
        hybrid_score  = 0.70 * xgb_prob + 0.30 * mse_norm

        prediction = 1 if hybrid_score >= hybrid_threshold else 0
        result     = "⚠️ FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE TRANSACTION"

        # 5. Generate SHAP explanation chart
        try:
            shap_img = generate_shap_chart(feature_values)
        except Exception:
            shap_img = None

        # 6. Generate graph neighbourhood chart
        try:
            graph_img = generate_graph_chart(location, feature_values)
        except Exception:
            graph_img = None

        # 7. Logging
        row = dict(feature_values)
        row['Fraudulent']        = int(prediction)
        row['Fraud_Probability'] = round(float(hybrid_score), 4)
        row['Timestamp']         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_exists = os.path.isfile(DYNAMIC_DATA_PATH)
        with open(DYNAMIC_DATA_PATH, 'a', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return render_template("home.html",
                               result=result,
                               probability=round(float(hybrid_score) * 100, 2),
                               shap_img=shap_img,
                               graph_img=graph_img)

    except Exception as e:
        return render_template("home.html", error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
