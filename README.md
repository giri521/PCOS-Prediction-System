# PCOS Prediction System

[Live Demo](https://pcos-prediction-system.onrender.com/) 🌐

---

## 🩺 About

A web-based **machine learning system** to predict the risk of **Polycystic Ovary Syndrome (PCOS)** from clinical, hormonal, and lifestyle data.

It integrates **three trained ML models**, each built using a different feature selection strategy:

* Mutual Information (MI)
* T-Test / ANOVA F-Score
* Combined (Union of both)

---

## ✨ Features

* Collects user inputs: Age, Weight, Height, BMI, Hormone levels (FSH, LH, AMH, PRL, TSH), Cycle details, Follicle counts, Lifestyle indicators.
* Auto-calculates **FSH/LH ratio**.
* Runs real-time predictions using Flask backend.
* Displays PCOS Risk levels (Low / Moderate / High) with personalized health advice.
* Deployed online using Render.

---

## 🧩 Project Structure

```
├── dataset/               ← Original dataset
├── models/                ← Saved models and feature files
├── templates/             ← HTML pages for Flask app
├── app.py                 ← Flask web application
├── train.py               ← Model training & feature selection script
├── requirements.txt       ← Python dependencies
└── README.md              ← Project documentation
```

---

## ⚙️ How It Works

1. Collects input from the web form.
2. Calculates the FSH/LH ratio automatically.
3. Loads the correct model and scaler.
4. Selects appropriate features and scales input.
5. Predicts PCOS probability.
6. Displays result with confidence score and health tips.

---

## 🧠 Model Training Pipeline (`train.py`)

* Loads and cleans the dataset.
* Fills missing values with medians.
* Encodes Yes/No responses as 1/0.
* Scales features using `StandardScaler`.
* Performs feature selection using:

  * Mutual Information (top 10 features)
  * ANOVA / T-Test (top 10 features)
  * Combined = union of both sets.
* Trains three Bagging-RandomForest models.
* Saves all models and feature lists in the `/models/` folder.

### Model Performance

| Model              | Accuracy | F1-Score |
| ------------------ | -------- | -------- |
| Mutual Information | ~88%     | ~85%     |
| T-Test / ANOVA     | ~86%     | ~82%     |
| Combined           | ~90%     | ~87%     |

---

## 🌐 Deployment (Render)

* Flask app listens on `0.0.0.0` and dynamically binds to the Render-assigned port (`os.environ['PORT']`).
* All models and HTML templates are bundled into the deployment.
* Access the live app: [https://pcos-prediction-system.onrender.com/](https://pcos-prediction-system.onrender.com/)

### `requirements.txt`

```
Flask
pandas
numpy
joblib
scikit-learn
gunicorn
```

---

## 🚀 Getting Started

### Local Setup

```bash
git clone https://github.com/giri521/PCOS-Prediction-System.git
cd PCOS-Prediction-System
pip install -r requirements.txt
```

### Train Models (Optional)

```bash
python train.py
```

### Run Flask App

```bash
python app.py
```

Access the app at `http://127.0.0.1:5000`

---

## 🧮 Usage

1. Choose a model (MI, T-Test, or Combined).
2. Enter clinical and lifestyle data.
3. Submit the form.
4. View prediction results and confidence.
5. Check personalized health tips.

---
