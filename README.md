# 🚲 Bike Sharing Demand Prediction

## 📌 Project Overview

This project aims to predict bike-sharing demand using machine learning techniques. It uses CatBoost models and various features, including weather conditions and school holidays, to estimate bike rental demand.

## 🚀 Try the Live App

Want to see the application in action?

👉 **[Click here to open the live application](https://app-app-prediction-q9pktrdwottfwhmurtqcsn.streamlit.app/)**

## 📂 Repository Structure

```text
├── .gitignore
├── README.md
├── app_cloud_db.py
├── catboost_categories_spline.pkl
├── catboost_model_spline.pkl
├── functions_cloud_db.py
├── requirements.txt
└── school_holidays.py
```

## 📄 File Descriptions

| File | Description |
|---|---|
| `.gitignore` | Specifies files and directories that Git should ignore. |
| `README.md` | Contains project documentation and setup instructions. |
| `app_cloud_db.py` | Main Streamlit application for bike-sharing demand prediction with cloud database integration. |
| `catboost_categories_spline.pkl` | Saved CatBoost model for categorical features. |
| `catboost_model_spline.pkl` | Saved CatBoost model using spline-based features. |
| `functions_cloud_db.py` | Contains helper functions for data processing, weather data retrieval, database operations, and predictions. |
| `requirements.txt` | Lists the Python dependencies required to run the project. |
| `school_holidays.py` | Contains functionality for handling school holiday information. |

## 🛠️ Technologies Used

- Python
- Streamlit
- CatBoost
- Pandas
- Scikit-learn
- MySQL
- OpenWeatherMap API

## ⚙️ Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure secrets**

Set up the required database credentials and API keys using Streamlit Secrets.

**4. Run the application**

```bash
streamlit run app_cloud_db.py
```

## 🔐 Configuration

The application may require:
- Database connection credentials.
- An OpenWeatherMap API key.

For security reasons, sensitive credentials and API keys should never be committed to the GitHub repository.

## 🤝 Collaboration

This project supports collaborative development through GitHub.

Contributors can:
- Create their own forks.
- Work independently on their copies.
- Submit Pull Requests to propose changes.
- Review and merge approved contributions into the main repository.

## 📜 License

No license has been specified yet.

## 📜 License

No license has been specified yet.
