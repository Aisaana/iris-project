import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загружаем признаки и метки
with open('models/features.pkl', 'rb') as f:
    data = pickle.load(f)
    features = data['features']
    labels = data['labels']

print(f"Загружено {len(features)} образцов.")

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оцениваем на тестовой выборке
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность на тестовой выборке: {accuracy * 100:.2f}%")

# Сохраняем модель
with open('models/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Модель сохранена в models/iris_model.pkl")