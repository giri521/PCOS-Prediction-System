# ==============================================================
# 🩺 PCOS Prediction Flask App (3 Model Integration)
# Updated: FSH/LH ratio correction (auto-calculated)
# ==============================================================

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import joblib
import sys

app = Flask(__name__)
app.secret_key = 'super_secret_pcos_key'

# --- Model Metrics (hardcoded for display) ---
METRICS = {
    'mi': {'Accuracy': 88.0, 'F1-Score': 85.0, 'set_name': 'Mutual Information', 'endpoint': 'mutual_info_page'},
    'ttest': {'Accuracy': 86.0, 'F1-Score': 82.0, 'set_name': 'T-Test (ANOVA F-Score)', 'endpoint': 'ttest_page'},
    'combined': {'Accuracy': 90.0, 'F1-Score': 87.0, 'set_name': 'Combined (MI ∪ T-Test)', 'endpoint': 'combined_page'}
}

# --- Universal Input Features ---
UNIVERSAL_FEATURES = [
    'Age (yrs)', 'Weight (Kg)', 'Height (Cm)', 'BMI', 'Cycle(R/I)',
    'Cycle length(days)', 'FSH (mIU/ml)', 'LH (mIU/ml)', 'FSH/LH',
    'AMH (ng/ml)', 'PRL (ng/ml)', 'TSH (mIU/L)', 'Fast food (Y/N)',
    'Weight gain (Y/N)', 'Hair growth (Y/N)', 'Skin darkening (Y/N)',
    'Pimples (Y/N)', 'Ls', 'Rs'
]

# --- Feature Labels, Hints & Inputs ---
FEATURE_LABELS = {
    'Age (yrs)': {'label': 'Age (years)', 'hint': 'Range: 10 - 50', 'input_type': 'number', 'min_val': 10, 'max_val': 50},
    'Weight (Kg)': {'label': 'Weight (Kg)', 'hint': 'Range: 30 - 200', 'input_type': 'number', 'min_val': 30, 'max_val': 200},
    'Height (Cm)': {'label': 'Height (Cm)', 'hint': 'Range: 100 - 220', 'input_type': 'number', 'min_val': 100, 'max_val': 220},
    'BMI': {'label': 'Body Mass Index', 'hint': 'Range: 15 - 60 (kg/m²)', 'input_type': 'number', 'min_val': 15, 'max_val': 60, 'step': 0.1},
    'Cycle length(days)': {'label': 'Cycle Length (days)', 'hint': 'Range: 2 - 90', 'input_type': 'number', 'min_val': 15, 'max_val': 90},
    'FSH (mIU/ml)': {'label': 'FSH (mIU/mL)', 'hint': 'Enter Follicle-Stimulating Hormone level', 'input_type': 'number', 'min_val': 0, 'max_val': 100, 'step': 0.01},
    'LH (mIU/ml)': {'label': 'LH (mIU/mL)', 'hint': 'Enter Luteinizing Hormone level', 'input_type': 'number', 'min_val': 0, 'max_val': 100, 'step': 0.01},
    'FSH/LH': {'label': 'FSH/LH Ratio', 'hint': 'Auto-calculated or manually enter (FSH ÷ LH)', 'input_type': 'number', 'min_val': 0.1, 'max_val': 5.0, 'step': 0.01},
    'AMH (ng/ml)': {'label': 'AMH (ng/mL)', 'hint': 'Range: 0 - 20', 'input_type': 'number', 'min_val': 0, 'max_val': 20, 'step': 0.01},
    'PRL (ng/ml)': {'label': 'Prolactin (ng/mL)', 'hint': 'Range: 0 - 150', 'input_type': 'number', 'min_val': 0, 'max_val': 150, 'step': 0.01},
    'TSH (mIU/L)': {'label': 'Thyroid Stimulating Hormone', 'hint': 'Range: 0 - 50', 'input_type': 'number', 'min_val': 0, 'max_val': 50, 'step': 0.01},
    'Rs': {'label': 'Follicle No. (Right Ovary)', 'hint': 'Range: 0 - 50', 'input_type': 'number', 'min_val': 0, 'max_val': 50},
    'Ls': {'label': 'Follicle No. (Left Ovary)', 'hint': 'Range: 0 - 50', 'input_type': 'number', 'min_val': 0, 'max_val': 50},
    'Cycle(R/I)': {'label': 'Menstrual Cycle Regularity', 'hint': 'Select cycle status', 'input_type': 'select', 'options': {1.0: '1 = Irregular', 2.0: '2 = Regular'}},
    'Fast food (Y/N)': {'label': 'Frequent Fast Food Consumption', 'hint': 'Select consumption frequency', 'input_type': 'select_yn'},
    'Weight gain (Y/N)': {'label': 'Weight Gain', 'hint': 'Select presence of weight gain', 'input_type': 'select_yn'},
    'Hair growth (Y/N)': {'label': 'Hair Growth (Hirsutism)', 'hint': 'Select abnormal hair growth', 'input_type': 'select_yn'},
    'Skin darkening (Y/N)': {'label': 'Skin Darkening', 'hint': 'Select presence of skin darkening', 'input_type': 'select_yn'},
    'Pimples (Y/N)': {'label': 'Pimples / Acne', 'hint': 'Select presence of acne', 'input_type': 'select_yn'}
}

# --- Load Models and Artifacts ---
try:
    mi_model = joblib.load("models/mi_model.joblib")
    ttest_model = joblib.load("models/ttest_model.joblib")
    combined_model = joblib.load("models/combined_model.joblib")

    scaler = joblib.load("models/scaler.joblib")
    full_columns = joblib.load("models/full_columns.joblib")

    mi_features = joblib.load("models/mi_features.joblib")
    t_features = joblib.load("models/t_features.joblib")
    combined_features = joblib.load("models/combined_features.joblib")

except Exception as e:
    print(f"❌ Error loading model files: {e}")
    sys.exit(1)

# --- Prediction Function ---
def predict_input(user_data, route_key):
    input_df = pd.DataFrame(user_data, index=[0])
    final_df = pd.DataFrame(0.0, index=[0], columns=full_columns)

    for col in input_df.columns:
        if col in final_df.columns:
            final_df[col] = input_df[col].iloc[0]

    scaled = scaler.transform(final_df)
    scaled_df = pd.DataFrame(scaled, columns=full_columns)

    if route_key == 'mi':
        features, model = mi_features, mi_model
    elif route_key == 'ttest':
        features, model = t_features, ttest_model
    else:
        features, model = combined_features, combined_model

    scaled_input = scaled_df[features]
    prob = model.predict_proba(scaled_input)[0][1] * 100

    if prob >= 40.0:
        return "🩺 PCOS Detected (High Sensitivity)", prob, prob
    else:
        return "✅ No PCOS", prob, 100.0 - prob

# --- Risk-based Tips ---
def generate_tips(risk):
    if risk >= 60.0:
        return {'level': 'High Risk', 'color': 'border-red-600',
                'tips': [
                    "Seek immediate consultation with an endocrinologist or gynecologist.",
                    "Request hormonal and metabolic tests.",
                    "Adopt a strict low-carb diet and regular exercise.",
                    "Discuss medical options for managing symptoms."
                ]}
    elif risk >= 20.0:
        return {'level': 'Moderate Risk', 'color': 'border-yellow-600',
                'tips': [
                    "Consult a specialist for confirmation.",
                    "Keep track of menstrual cycle and symptoms.",
                    "Focus on low-glycemic diet and exercise.",
                    "Consider insulin resistance screening."
                ]}
    else:
        return {'level': 'Low Risk', 'color': 'border-green-600',
                'tips': [
                    "Maintain a healthy lifestyle and diet.",
                    "Do regular check-ups to monitor hormones.",
                    "Stay informed about early PCOS symptoms.",
                    "Include Vitamin D and Omega-3 in your diet."
                ]}

# --- Route Handlers ---
@app.route('/')
def index():
    return render_template("index_home.html")

def handle_prediction_route(route_key):
    feature_list_to_render = UNIVERSAL_FEATURES
    page_title = METRICS[route_key]['set_name']
    endpoint = METRICS[route_key]['endpoint']

    labeled_features = [{
        'raw_name': f,
        'label': FEATURE_LABELS.get(f, {}).get('label', f),
        'hint': FEATURE_LABELS.get(f, {}).get('hint', 'Enter value.'),
        'input_type': FEATURE_LABELS.get(f, {}).get('input_type', 'number'),
        'options': FEATURE_LABELS.get(f, {}).get('options', {}),
        'min_val': FEATURE_LABELS.get(f, {}).get('min_val', None),
        'max_val': FEATURE_LABELS.get(f, {}).get('max_val', None),
        'step': FEATURE_LABELS.get(f, {}).get('step', 1)
    } for f in feature_list_to_render]

    if request.method == 'POST':
        try:
            inputs = {f: float(request.form.get(f, 0.0)) for f in feature_list_to_render}

            # 🔹 Auto-calculate FSH/LH ratio safely
            if inputs.get('LH (mIU/ml)', 0) != 0:
                inputs['FSH/LH'] = inputs['FSH (mIU/ml)'] / inputs['LH (mIU/ml)']
            else:
                inputs['FSH/LH'] = 0.0

            result_text, pcos_risk, confidence = predict_input(inputs, route_key)

            friendly_inputs = {}
            for f in feature_list_to_render:
                label = FEATURE_LABELS.get(f, {}).get('label', f)
                value = inputs[f]
                input_type = FEATURE_LABELS.get(f, {}).get('input_type', 'number')

                if input_type == 'select_yn':
                    friendly_inputs[label] = 'Yes' if value == 1.0 else 'No'
                elif input_type == 'select':
                    options = FEATURE_LABELS.get(f, {}).get('options', {})
                    display_value = options.get(value, str(value))
                    friendly_inputs[label] = display_value
                else:
                    friendly_inputs[label] = value

            session['prediction_data'] = {
                'result_text': result_text,
                'pcos_risk': pcos_risk,
                'confidence': f"{confidence:.2f}",
                'inputs': friendly_inputs,
                'route_key': route_key
            }

            return redirect(url_for('result_page'))

        except ValueError:
            prediction_text = "⚠️ Input error. Please ensure valid numeric inputs."
            return render_template("form.html",
                                   features=labeled_features,
                                   page_title=page_title,
                                   prediction_text=prediction_text,
                                   endpoint=endpoint)

    return render_template("form.html",
                           features=labeled_features,
                           page_title=page_title,
                           endpoint=endpoint)

@app.route('/mi', methods=['GET', 'POST'])
def mutual_info_page():
    return handle_prediction_route('mi')

@app.route('/ttest', methods=['GET', 'POST'])
def ttest_page():
    return handle_prediction_route('ttest')

@app.route('/combined', methods=['GET', 'POST'])
def combined_page():
    return handle_prediction_route('combined')

@app.route('/result')
def result_page():
    data = session.pop('prediction_data', None)
    if data is None:
        return redirect(url_for('index'))

    route_key = data['route_key']
    metrics = METRICS.get(route_key, {})
    pcos_risk = data['pcos_risk']
    data['risk_info'] = generate_tips(pcos_risk)

    return render_template("result.html", data=data, metrics=metrics)

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
