import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv('student_data.csv')

# Encode categoricals
le_gender = LabelEncoder()
le_parent = LabelEncoder()
df['gender_enc'] = le_gender.fit_transform(df['gender'])
df['parent_enc'] = le_parent.fit_transform(df['parental_education'])

features = ['study_hours_per_day','attendance_percentage','previous_score',
            'assignments_completed','sleep_hours','math_score','science_score',
            'english_score','extra_curricular','internet_access','gender_enc','parent_enc']

X = df[features]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop Features:")
print(importances.head(6))

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'le_gender': le_gender, 'le_parent': le_parent, 'features': features}, f)

print("\nModel saved!")
