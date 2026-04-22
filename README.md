# 🏥 Medical AI Platform

A full-stack AI-powered medical dashboard built using Flask, Machine Learning, and NLP — designed to simulate real-world clinical software systems.

---

## 🚀 Live Demo
👉 https://your-render-url.onrender.com

---

## 🧠 Overview

This project is a **complete medical analytics platform** that integrates:

- Patient management system
- Diabetes risk prediction (ML)
- Mood tracking with sentiment analysis (NLP)
- Medical image classification (demo pipeline)
- Drug interaction checker
- Clinical dashboard with real-time insights

Built with a focus on **clean UI, real-world workflows, and deployability**.

---

## ⚙️ Tech Stack

| Layer       | Technology |
|------------|-----------|
| Backend     | Flask (Python) |
| Frontend    | HTML, CSS, Vanilla JS |
| Templates   | Jinja2 |
| Database    | SQLite |
| ML Models   | scikit-learn (Random Forest, Logistic Regression) |
| NLP         | TextBlob |
| Charts      | Chart.js |
| Deployment  | Render |
| Styling     | Custom CSS (Dark Medical UI) |

---

## 📁 Project Structure
medical_platform/
│
├── app.py
├── db.py
├── ml.py
├── requirements.txt
├── Procfile
│
├── templates/
│ ├── base.html
│ ├── dashboard.html
│ ├── patients.html
│ ├── diabetes.html
│ ├── mood.html
│ ├── image.html
│ ├── drugs.html
│ └── history.html
│
└── static/
├── style.css
└── app.js
---

## ✨ Features

### 👨‍⚕️ Patient Management
- Add, delete, and view patient records
- Structured relational database

---

### 🩺 Diabetes Prediction
- Choose model:
  - Random Forest
  - Logistic Regression
- Real-time prediction with:
  - Risk level (Low / High)
  - Confidence score
- Feature importance & coefficient visualization

---

### 💬 Mood Tracker (NLP)
- Journal-based mood logging
- Sentiment analysis using TextBlob
- Outputs:
  - Sentiment label (Positive / Neutral / Negative)
  - Polarity score
- Visual insights:
  - Mood trend chart
  - Sentiment distribution
  - Keyword analysis

---

### 🧠 Image Classification (Demo)
- Upload medical images:
  - X-ray
  - MRI
  - Skin lesion
  - Retina
- Returns top 3 predictions with confidence

---

### 💊 Drug Interaction Checker
- Detects harmful drug combinations
- Severity levels:
  - Severe
  - Moderate
  - Safe
- Clinical guidance output

---

### 📊 Dashboard
- Key metrics:
  - Patients
  - Diabetes checks
  - Mood entries
  - Image scans
  - Drug checks
- Charts:
  - Diabetes risk distribution (Pie)
  - Mood trends (Line)
- Recent activity table

---

### 📁 Patient History
- Unified timeline for:
  - Diabetes records
  - Mood entries
  - Image scans
  - Drug checks
- CSV export support

---

## 🎨 UI Design

- Dark clinical theme
- Data-dense dashboards
- Color-coded risk indicators:
  - 🟢 Low Risk
  - 🟡 Moderate
  - 🔴 High Risk
- Responsive layout (mobile + desktop)
- AJAX-based interactions (no page reloads)

---

## 🛠️ Installation (Local Setup)

### 1. Clone the repo
```bash
git clone https://github.com/Adithya2005-spec/medical-platform.git
cd medical-platform
### 2.Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
### 3.Install Dependencies
pip install -r requirements.txt
### 4.Download NLP Data
python -m textblob.download_corpora
### 5.Run the app
python app.py
### 6.Open in Browser
http://127.0.0.1:5000
----

### ☁️ Deployment (Render)
Required Files:
requirements.txt
Procfile

### Procfile
web: gunicorn app:app

### Render Settings:
Setting	                                 Value
Build Command	                           pip install -r requirements.txt
Start Command	                           gunicorn app:app

⚠️ Notes
SQLite database resets on redeploy (Render free tier)
ML models auto-train if .pkl not found
Image classification is a demo pipeline (no heavy DL model)
🧪 Future Improvements
PostgreSQL integration (persistent DB)
User authentication system
Real deep learning model (CNN)
API endpoints for mobile apps
Doctor dashboard with alerts
🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

📜 License

This project is for educational and demonstration purposes.

👨‍💻 Author

Adithya S Nayak 

⭐ If you like this project

Give it a star ⭐ on GitHub and share your feedback!


---

If you want to make it even stronger:
- I can add **badges (build, deploy, tech stack)**  
- Or create a **top-tier README with screenshots + GIF demo**
