# 🐨 Possum Age Prediction — KNN Regression

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that predicts the **age of a Mountain Brushtail Possum** (*Trichosurus caninus*) from its body measurements using a **K-Nearest Neighbors Regressor**, deployed as an interactive Streamlit web app.

---

## 📌 Problem Statement

Estimating a possum's age in the field typically requires invasive procedures. This project explores whether simple, non-invasive body measurements (head length, skull width, body length, etc.) can be used to estimate age with reasonable accuracy — useful for wildlife researchers and conservationists.

---

## 🗂️ Project Structure

```
possum-age-predictor/
├── app.py                            # Streamlit web application
├── knn_possum_pipeline.pkl           # Trained pipeline (preprocessing + model)
├── Possum_Age_Prediction_with_KNN.ipynb  # Full notebook with EDA, training, evaluation
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🔬 Dataset

| Property | Value |
|---|---|
| Source | Possum Regression Dataset |
| Records | 104 (102 after dropping missing targets) |
| Features | 11 body measurements + 2 categorical |
| Target | Age in years (1–9) |

**Features used:** `Pop`, `sex`, `hdlngth`, `skullw`, `totlngth`, `taill`, `footlgth`, `earconch`, `eye`, `chest`, `belly`

**Dropped:** `case` (row ID), `site` (geographic ID redundant with `Pop`)

---

## ⚙️ ML Pipeline

```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num_pipeline  →  SimpleImputer(median)  →  StandardScaler
│   └── cat_pipeline  →  OneHotEncoder(handle_unknown='ignore')
└── KNeighborsRegressor
```

Hyperparameters tuned with **GridSearchCV** (5-fold CV):

| Parameter | Search Space |
|---|---|
| `n_neighbors` | 3, 5, 7, 9, 11, 13, 15, 17, 19 |
| `weights` | uniform, distance |
| `metric` | euclidean, manhattan |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| MAE | 1.4942 years |
| MSE | 3.2587 |
| RMSE | 1.8052 years |
| R² Score | 0.2872 |

> The R² of 0.29 reflects a modest but expected result given the small dataset (102 rows), weak feature-target correlations (max 0.36), and scarcity of older possum samples (ages 7–9).

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/NoureldeenBassem/possum-age-predictor.git
cd possum-age-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **Note:** The `knn_possum_pipeline.pkl` file must be in the same directory as `app.py`. Generate it by running the notebook end-to-end, or download it from the releases.

---

## 🌐 Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select this repo → set **Main file path** to `app.py`.
4. Click **Deploy**.

Streamlit Cloud installs `requirements.txt` automatically.

---

## 🔍 Key Findings from EDA

- Age 3 is the most common (27 possums); ages 8–9 are rare (1–2 each).
- `belly` (0.36), `chest` (0.34), and `hdlngth` (0.33) show the strongest correlations with age.
- Many body measurements are highly intercorrelated (e.g. `hdlngth` & `skullw` = 0.71), which is expected as body dimensions scale together.
- No single feature is a strong predictor of age — the non-parametric, multi-feature nature of KNN is well suited to this problem.

---

## 🛠️ Future Improvements

- Feature engineering: ratios of body measurements (e.g. head-to-body ratio)
- Compare against Random Forest and Gradient Boosting regressors
- Collect more data for older possums (ages 7–9) to improve generalization

---

## 👤 Author

**Noureldeen Bassem**  
Computer & Intelligent Systems Engineering — Future University in Egypt  
[GitHub](https://github.com/NoureldeenBassem) · [LinkedIn](https://www.linkedin.com/in/noureldeen-bassem)
