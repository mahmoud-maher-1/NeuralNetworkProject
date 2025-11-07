# Neural Network Classifier Project

## Authors  
- **Akram ElNahtawy**  
- **Bassam Hassan**  
- **Mahmoud Maher**  
- **Mai Farahat**  
- **Mohanad Sabry**

---

## Try our live deployment at https://neuralnetworkproject.streamlit.app/

---

## 1. Project Overview
An interactive Streamlit application that trains, evaluates and visualises two classic linear neural-network classifiers—**Perceptron** and **Adaline (Gradient-Descent)**—on a user-selected pair of bird-species features.  
The tool guides users from data selection → hyper-parameter tuning → live training → performance metrics and decision-boundary plot in a single click.

---

## 2. System Requirements
- Python 3.9+  
- 200 MB free disk space (mostly for Python env & browser cache)  
- Internet connection (only during initial pip install; app runs fully offline afterwards)

---

## 3. Installation Guide

### 3.1 Clone / download
```bash
git clone <repository-url>  # or unzip the delivered folder
cd neural-network-classifier
```

### 3.2 Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3.3 Install dependencies
```bash
pip install -r requirements.txt
```
*The pinned versions are listed in* `requirements.txt`.  
*No extra system libraries are required (pure-Python stack).*

### 3.4 (Optional) Prepare your data
Place the raw CSV file at  
`data/dataset.csv`  
with at least the columns  
`bird_category, gender, body_mass, beak_length, beak_depth, fin_length`.  
*If the file is missing the app will still start, but training will fail with a clear message.*

---

## 4. How to Run
From the project root (with venv activated):
```bash
streamlit run app.py
```
Your default browser should open `http://localhost:8501`.  
Use the side-bar or the in-app **"Train a New Model"** button to navigate.

---

## 5. Usage Walk-through
1. **Home page**  
   - Pick two different features (X- and Y-axis).  
   - Pick two different bird classes (A, B, C).  
   - Choose algorithm: Perceptron or Adaline.  
   - Adjust epochs, learning-rate, optional MSE early-stop, bias toggle.  
   - Press **"Train Model"**.

2. **Results page**  
   - Confusion matrix, accuracy, precision, recall, F1.  
   - Matplotlib figure with train/test scatter and red dashed decision boundary.  
   - **"Train a New Model"** resets the session and returns you to Home.

---

## 6. Codebase Map
```
.
├── app.py                       # Entry-point, session-state router
├── requirements.txt
├── data/
│   ├── dataset.csv              # raw (user-supplied)
│   └── processed.csv            # auto-generated cache
├── ui/
│   ├── home_page.py             # Streamlit form & orchestration
│   └── results_page.py          # metrics table, CM, figure display
├── core/
│   ├── training.py              # Perceptron & AdalineGD + factory
│   ├── preprocessing.py         # clean, impute, StandardScaler, split
│   ├── prediction.py            # linear forward pass → ±1
│   ├── evaluation.py            # CM, acc, prec, rec, F1
│   └── plotting.py              # matplotlib decision-boundary figure
├── models/
│   └── validation_models.py     # Pydantic schemas for type safety
└── utils/
    └── constants.py             # defaults & file paths
```

---

## 7. Key Algorithms
### Perceptron
- Rosenblatt update rule, signum activation.  
- Stops after fixed epochs; tracks #misclassifications per epoch.

### Adaline (Batch Gradient Descent)
- Linear activation, minimises MSE.  
- Optional early-stop when MSE ≤ threshold.  
- Updates weights once per epoch via full gradient.

Both classifiers map the chosen pair of features into **-1** vs **+1** labels.

---

## 8. Licence
MIT – feel free to use, modify and distribute with attribution.

---

**Enjoy experimenting with linear neural networks!**
