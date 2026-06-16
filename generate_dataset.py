import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

study_hours = np.random.normal(4, 2.5, n).clip(0, 12)
attendance = np.random.normal(68, 20, n).clip(20, 100)
prev_score = np.random.normal(55, 18, n).clip(10, 100)
assignments = np.random.randint(0, 11, n)
sleep_hours = np.random.normal(6.5, 1.5, n).clip(3, 10)
math = np.random.normal(55, 20, n).clip(0, 100)
science = np.random.normal(57, 18, n).clip(0, 100)
english = np.random.normal(60, 16, n).clip(0, 100)
extra_curricular = np.random.choice([0, 1], n, p=[0.4, 0.6])
internet_access = np.random.choice([0, 1], n, p=[0.3, 0.7])
gender = np.random.choice(['Male', 'Female'], n)
parental_education = np.random.choice(['None', 'High School', 'Graduate', 'Post-Graduate'], n)

score = (
    study_hours * 3.5 +
    attendance * 0.4 +
    prev_score * 0.3 +
    assignments * 1.8 +
    sleep_hours * 0.8 +
    (math + science + english) / 3 * 0.15 +
    extra_curricular * 2 +
    internet_access * 3 +
    np.random.normal(0, 8, n)
)

score = ((score - score.min()) / (score.max() - score.min())) * 80 + 10
score = score.clip(0, 100)

final_grade = pd.cut(score, bins=[0,40,55,65,80,100], labels=['F','D','C','B','A'])
result = (score >= 40).astype(int)

df = pd.DataFrame({
    'gender': gender,
    'parental_education': parental_education,
    'internet_access': internet_access,
    'study_hours_per_day': study_hours.round(1),
    'attendance_percentage': attendance.round(1),
    'previous_score': prev_score.round(1),
    'assignments_completed': assignments,
    'sleep_hours': sleep_hours.round(1),
    'math_score': math.round(1),
    'science_score': science.round(1),
    'english_score': english.round(1),
    'extra_curricular': extra_curricular,
    'final_score': score.round(1),
    'grade': final_grade,
    'result': result
})

df.to_csv('student_data.csv', index=False)
print("Dataset generated:", df.shape)
print(df['grade'].value_counts())
print(f"Pass rate: {df['result'].mean()*100:.1f}%")
