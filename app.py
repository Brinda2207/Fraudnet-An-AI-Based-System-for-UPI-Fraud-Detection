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
try:
    rf_model = joblib.load('models/rf_model.pkl')
except FileNotFoundError:
    rf_model = None

# Transaction CSV loaded lazily inside generate_graph_chart() — doesn't exist on first deploy

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
def generate_shap_chart(feature_values_dict, is_fraud):
    """
    feature_values_dict : raw (unscaled) feature dict from the form.
    is_fraud            : bool — whether prediction is fraud or legit.
    Returns             : (base64_png, explanation_text) tuple, or (None, None).
    """
    if rf_model is None:
        return None, None

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

    input_array  = np.array([feature_values_dict[f] for f in features]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_df     = pd.DataFrame(input_scaled, columns=[label_map.get(f, f) for f in features])

    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(input_df)

    # shape (1, 12, 2) — class 1 = fraud
    sv = shap_values[0][:, 1]
    ev = explainer.expected_value[1]

    explanation = shap.Explanation(
        values        = sv,
        base_values   = ev,
        data          = input_df.iloc[0].values,
        feature_names = input_df.columns.tolist()
    )

    # ── smaller chart ────────────────────────────────────────────────────────
    plt.style.use('dark_background')
    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor('#0f172a')
    fig.set_size_inches(5.5, 3.8)   # was (8,5) — now smaller
    img_b64 = fig_to_base64(fig)

    # ── build human-readable text explanation ────────────────────────────────
    # pair feature names with their shap values (class 1 = fraud direction)
    feat_names = input_df.columns.tolist()
    pairs = sorted(zip(sv, feat_names), key=lambda x: abs(x[0]), reverse=True)

    # top 3 features pushing toward fraud (positive sv) and toward legit (negative sv)
    toward_fraud = [(n, v) for v, n in pairs if v > 0][:3]
    toward_legit = [(n, v) for v, n in pairs if v < 0][:3]

    if is_fraud:
        verdict = "flagged as FRAUD"
        primary = toward_fraud if toward_fraud else toward_legit
        direction_word = "increased"
    else:
        verdict = "classified as LEGITIMATE"
        primary = toward_legit if toward_legit else toward_fraud
        direction_word = "reduced"

    if primary:
        top_feature  = primary[0][0]
        second_parts = [f"<b>{n}</b>" for n, v in primary[1:3]]
        others = (", along with " + " and ".join(second_parts)) if second_parts else ""
        text = (f"This transaction was {verdict} mainly because <b>{top_feature}</b> "
                f"{direction_word} the fraud risk{others}. "
                f"Each bar shows how much a feature pushed the prediction toward fraud (red) "
                f"or toward legitimate (blue).")
    else:
        text = f"This transaction was {verdict}. See the chart for feature-level breakdown."

    return img_b64, text


# ── Graph neighbourhood chart ─────────────────────────────────────────────────
def generate_graph_chart(location_val, feature_values_dict, prediction):
    """
    Builds a transaction network graph.
    - If dynamic_transactions.csv exists (has prior transactions): uses real logged data.
    - Otherwise: builds a synthetic illustrative graph around this transaction's features.
    Always shows the current transaction as a highlighted orange node.
    Returns (base64_png, explanation_text) or (None, None).
    """
    THIS_USER = 5000   # fixed User_ID used in app.py for all submissions

    loc_labels = {0: 'Local (same city)', 1: 'Domestic (diff city)', 2: 'International'}

    # ── Try to load real logged data ──────────────────────────────────────────
    if os.path.isfile('dynamic_transactions.csv'):
        try:
            df_full = pd.read_csv('dynamic_transactions.csv')
            required = ['User_ID','Location','Fraudulent',
                        'Number_of_Transactions_Last_24H','Previous_Fraudulent_Transactions']
            if all(c in df_full.columns for c in required) and len(df_full) >= 3:
                df_loc = df_full[df_full['Location'] == location_val].copy()
                if not df_loc.empty:
                    sample_users = list(df_loc['User_ID'].unique())[:60]
                    df_loc = df_loc[df_loc['User_ID'].isin(sample_users)]
                    fraud_set = set(df_loc[df_loc['Fraudulent'] == 1]['User_ID'].unique())
                    mule_set  = set(df_loc[
                        (df_loc['Number_of_Transactions_Last_24H'] > 10) &
                        (df_loc['Previous_Fraudulent_Transactions'] > 0)
                    ]['User_ID'].unique())
                    G = nx.Graph()
                    for u in df_loc['User_ID'].unique():
                        G.add_node(u)
                    users_at_loc = df_loc['User_ID'].unique()
                    for i in range(len(users_at_loc)):
                        for j in range(i+1, len(users_at_loc)):
                            G.add_edge(users_at_loc[i], users_at_loc[j])
                    data_source = f"Real logged transactions — {loc_labels.get(location_val,'')} network"
                    return _draw_graph(G, THIS_USER, fraud_set, mule_set,
                                       prediction, location_val, data_source)
        except Exception:
            pass   # fall through to synthetic

    # ── Synthetic graph — always works even on first transaction ──────────────
    np.random.seed(42)
    n_normal = 18
    n_fraud  = 4
    n_highrisk = 3

    normal_ids   = list(range(1000, 1000 + n_normal))
    fraud_ids    = list(range(2000, 2000 + n_fraud))
    highrisk_ids = list(range(3000, 3000 + n_highrisk))
    all_ids      = normal_ids + fraud_ids + highrisk_ids + [THIS_USER]

    G = nx.Graph()
    for uid in all_ids:
        G.add_node(uid)

    # Connect this user to a few normal and fraud nodes (shared location)
    for nid in np.random.choice(normal_ids, size=5, replace=False):
        G.add_edge(THIS_USER, nid)
    if prediction == 1:   # fraud prediction → connect to fraud cluster
        for fid in fraud_ids[:2]:
            G.add_edge(THIS_USER, fid)
    for fid in fraud_ids:
        for nid in np.random.choice(normal_ids, size=2, replace=False):
            G.add_edge(fid, nid)
    for hid in highrisk_ids:
        for fid in fraud_ids:
            G.add_edge(hid, fid)
        G.add_edge(hid, np.random.choice(normal_ids))
    for i in range(8):
        a, b = np.random.choice(normal_ids, size=2, replace=False)
        G.add_edge(a, b)

    fraud_set    = set(fraud_ids)
    mule_set     = set(highrisk_ids)
    data_source  = f"Illustrative network — {loc_labels.get(location_val,'')} (no prior transactions logged yet)"
    return _draw_graph(G, THIS_USER, fraud_set, mule_set,
                       prediction, location_val, data_source)


def _draw_graph(G, this_user, fraud_set, mule_set, prediction, location_val, data_source):
    """Shared drawing logic for both real and synthetic graphs."""
    colors, sizes = [], []
    for node in G.nodes():
        if node == this_user:
            colors.append('#f59e0b')   # amber = THIS transaction
            sizes.append(400)
        elif node in mule_set:
            colors.append('#f39c12')   # orange = high-risk/mule
            sizes.append(250)
        elif node in fraud_set:
            colors.append('#ef4444')   # red = fraud
            sizes.append(150)
        else:
            colors.append('#3b82f6')   # blue = normal
            sizes.append(60)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax,
            node_color=colors, node_size=sizes,
            with_labels=False,
            edge_color='#334155', width=0.5, alpha=0.9)

    this_p   = mpatches.Patch(color='#f59e0b', label='This transaction')
    mule_p   = mpatches.Patch(color='#f39c12', label='High-risk account')
    fraud_p  = mpatches.Patch(color='#ef4444', label='Fraud account')
    normal_p = mpatches.Patch(color='#3b82f6', label='Normal account')
    ax.legend(handles=[this_p, mule_p, fraud_p, normal_p],
              loc='upper left', fontsize=8,
              facecolor='#1e293b', labelcolor='white', edgecolor='#334155')

    ax.set_title(data_source, fontsize=9, color='#94a3b8', pad=6)

    # ── build graph explanation text ─────────────────────────────────────────
    fraud_neighbours = sum(1 for n in G.neighbors(this_user) if n in fraud_set)
    mule_neighbours  = sum(1 for n in G.neighbors(this_user) if n in mule_set)
    total_neighbours = G.degree(this_user)

    if fraud_neighbours > 0 or mule_neighbours > 0:
        risk_txt = (f"This transaction's user shares a location with "
                    f"<b>{fraud_neighbours} fraud account(s)</b> and "
                    f"<b>{mule_neighbours} high-risk account(s)</b> "
                    f"out of {total_neighbours} nearby users — raising network-level suspicion.")
    else:
        risk_txt = (f"This transaction's user is connected to {total_neighbours} nearby users, "
                    f"none of whom have fraud history in this location — supporting a legitimate classification.")

    return fig_to_base64(fig), risk_txt


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
        shap_img, shap_text = None, None
        try:
            shap_img, shap_text = generate_shap_chart(feature_values, prediction == 1)
        except Exception:
            pass

        # 6. Generate graph neighbourhood chart
        graph_img, graph_text = None, None
        try:
            graph_img, graph_text = generate_graph_chart(location, feature_values, prediction)
        except Exception:
            pass

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
                               shap_text=shap_text,
                               graph_img=graph_img,
                               graph_text=graph_text)

    except Exception as e:
        return render_template("home.html", error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
