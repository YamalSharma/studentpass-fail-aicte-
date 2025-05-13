
# 🎓 Student Performance Prediction using Machine Learning

This project predicts whether a student will pass or fail based on personal, social, and academic features using a Machine Learning model. It includes both the model training pipeline and a deployed interactive web app using Streamlit.

---

## 📌 Project Overview

- **Objective**: Predict if a student will pass (G3 >= 10) or fail (G3 < 10) based on input features such as study time, absences, parental education, and support systems.
- **Dataset**: UCI Student Performance dataset (`student-mat.csv`)
- **Tech Stack**: Python, pandas, scikit-learn, Streamlit, pickle

---

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository
- **File**: `student-mat.csv` (Math course)
- **Records**: 395
- **Features**: 30+ (demographics, social, school-related)
- **Target Variable**: `pass` (1 = G3 ≥ 10, 0 = G3 < 10)

Key features used:
- `studytime`, `failures`, `absences`
- `Medu`, `Fedu` (parental education)
- `schoolsup`, `famsup` (extra educational support)
- `internet` (home internet access)

---

## ⚙️ How It Works

### 🔹 Model Training (`model_training.py`)
1. Loads dataset and applies label encoding to categorical columns.
2. Creates a binary `pass` column from final grade (`G3`).
3. Trains a Random Forest Classifier.
4. Saves the trained model as `student_model.pkl`.
5. Prints accuracy, classification report, and confusion matrix.

### 🔹 Web App (`app.py`)
1. Loads the trained model using `pickle`.
2. Takes user inputs through a clean Streamlit interface.
3. Predicts if the student will pass or fail.
4. Displays result as ✅ **Pass** or ❌ **Fail**.

---

## 🧠 Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Test Accuracy**: ~85% *(varies slightly depending on random split)*
- **Evaluation Metrics**:
    - Accuracy
    - Precision, Recall, F1-Score
    - Confusion Matrix

---

## 🚀 Running the Project

### 1. Clone the repository
```bash
git clone https://github.com/YamalSharma/studentpass-fail-aicte-.git
cd student-performance-ml
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn streamlit
```

### 3. Train the model
```bash
python train_mmodel.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---



## 📌 Example Predictions

| Study Time | Failures | Absences | Pass? |
|------------|----------|----------|-------|
| 2          | 0        | 5        | ✅ Yes |
| 1          | 3        | 20       | ❌ No |
| 4          | 0        | 0        | ✅ Yes |

---

## 📈 Feature Importance (Optional)

You can visualize feature importance using the following code inside `model_training.py`:

```python
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
```

---

## 👨‍💻 Author

- **Yamal Sharma**: [https://github.com/YamalSharma]

---

## 🏁 Future Improvements

- Use hyperparameter tuning with `GridSearchCV`
- Improve categorical encoding using One-Hot Encoding
- Add charts and explanations in the Streamlit dashboard
- Deploy to the web using Streamlit Cloud or Hugging Face Spaces

---

## 📁 Files in This Project

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for prediction |
| `train_model.py` | Model training, evaluation, and saving |
| `student_model.pkl` | Trained ML model |
| `student-mat.csv` | Dataset |


---

## 📜 License

This project is open-source and free to use under the MIT License.
