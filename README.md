# 🏠 House Price Prediction using Optuna & XGBoost

## 📌 Project Overview

This project predicts house prices based on various features such as area, location, number of rooms, and more.
It uses advanced machine learning techniques with **Optuna hyperparameter tuning** to select the best model.

---

## 🚀 Features

* 🔍 Automatic model selection using Optuna
* ⚡ Supports XGBoost & Gradient Boosting
* 📊 Data preprocessing with encoding & scaling
* 🌐 Interactive web app using Streamlit
* 💾 Model saving using Joblib

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Optuna
* Streamlit

---

## 📁 Project Structure

```
optuna-project/
│
├── app.py                 # Streamlit web app
├── train.py               # Model training script
├── requirements.txt
├── README.md
│
├── data/
│   └── House Price Prediction Dataset.csv
│
├── model/
│   └── house_price_model.joblib
```

---

## ⚙️ Installation

### 1️⃣ Clone repository

```
git clone https://github.com/your-username/optuna-project.git
cd optuna-project
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 🏋️ Train Model

```
python train.py
```

---

## 🌐 Run Application

```
streamlit run app.py
```

---

## 📊 Input Features

* Condition (Poor, Average, Good)
* Location (Rural, Suburban, Urban)
* Garage (Yes/No)
* Area (sq ft)
* Bedrooms
* Bathrooms
* Year Built
* Floors

---

## 🎯 Output

* Predicted house price 💰

---

## 📸 Demo

<img width="1545" height="721" alt="Screenshot 2026-04-29 121655" src="https://github.com/user-attachments/assets/8f63a96d-69e5-4301-ab69-0c8c6f35ac22" />


---

## 💡 Future Improvements

* Add feature importance visualization
* Deploy on cloud (Streamlit Cloud / Render)
* Improve UI/UX design
* Add real-time data integration

---

## 👨‍💻 Author

**Shanu Singh**

---

## ⭐ If you like this project

Give it a star on GitHub!
