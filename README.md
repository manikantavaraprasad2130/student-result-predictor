# 🎓 Student Result Predictor

An ML-powered web application that predicts student academic performance and identifies at-risk students using a Random Forest classification model.

**Live Demo:** [Deploy on Streamlit Cloud]  
**Built by:** Vanga Manikanta Varaprasad

---

## 📌 Features

- 🔮 **Predict** whether a student will pass or fail based on 12 input features
- 📊 **Analytics Dashboard** with grade distribution, attendance trends, and subject comparisons
- 📈 **Model Insights** — feature importance chart showing what factors matter most
- 💡 **Personalized Recommendations** for each student based on their inputs
- 🎯 **87% accuracy** using Random Forest with 1,000 student records

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Scikit-learn (Random Forest Classifier) |
| Data Processing | Pandas, NumPy |
| Frontend / UI | Streamlit |
| Visualizations | Plotly |
| Deployment | Streamlit Cloud |

---

## 📂 Project Structure

```
student-result-predictor/
│
├── app.py                  # Main Streamlit application
├── generate_dataset.py     # Script to generate synthetic student dataset
├── train_model.py          # ML model training script
├── student_data.csv        # Dataset (1000 student records)
├── model.pkl               # Trained Random Forest model
└── README.md
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/student-result-predictor.git
cd student-result-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (optional - model.pkl already included)
python train_model.py

# 4. Run the app
streamlit run app.py
```

---

## 📊 Dataset

- **1,000 synthetic student records** generated with realistic distributions
- **15 features** including study hours, attendance, subject scores, sleep, assignments
- **Target variable:** Pass (1) / Fail (0)
- **Pass rate:** ~78%

---

## 🤖 Model Details

- **Algorithm:** Random Forest Classifier (100 trees)
- **Train/Test Split:** 80/20
- **Accuracy:** 87%
- **Top Predictors:** Attendance %, Study Hours, Previous Score, Assignments Completed

---

## 💡 Key Insights

1. **Attendance** is the strongest predictor of academic success
2. **Study hours** directly correlates with final score
3. **Early scores** predict future performance — early intervention is crucial
4. **Assignment completion** is a behavioral indicator of student engagement

---

## 📋 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
```

---

## 👤 Author

**Vanga Manikanta Varaprasad**  
B.Tech CSE (AI & ML) | SR University, Warangal  
📧 manikantavaraprasadvanga00@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/manikanta-varaprasadvanga)
