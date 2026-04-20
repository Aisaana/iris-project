import cv2
import numpy as np
import pickle
from skimage.feature import hog, local_binary_pattern

# Загрузка каскадов для обнаружения лица и глаз
face_cascade = cv2.CascadeClassifier('models/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('models/cascades/haarcascade_eye.xml')

# Загрузка модели
with open('models/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

def extract_features_from_roi(roi):
    """Извлекаем признаки из региона глаза"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (60, 36))
    
    # Нормализация освещения
    gray = cv2.equalizeHist(gray)
    
    hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                   visualize=False, feature_vector=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    
    return np.concatenate([hog_feat, lbp_hist]).reshape(1, -1)

# Запуск камеры
cap = cv2.VideoCapture(0)
print("🎥 Распознавание запущено (Q — выход)")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    status = "No face detected"
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        
        # Ищем глаза в пределах лица
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) > 0:
            (ex, ey, ew, eh) = eyes[0]
            
            # Координаты глаза относительно всего фрейма
            eye_x = x + ex
            eye_y = y + ey
            
            # Выделяем регион глаза
            size = 50
            x1 = max(0, eye_x - size)
            y1 = max(0, eye_y - size)
            x2 = min(frame.shape[1], eye_x + ew + size)
            y2 = min(frame.shape[0], eye_y + eh + size)
            
            eye_roi = frame[y1:y2, x1:x2]
            
            if eye_roi.size > 0:
                # Предсказание
                features = extract_features_from_roi(eye_roi)
                prediction = model.predict(features)[0]
                confidence = max(model.predict_proba(features)[0])
                
                status = f"Person {prediction} ({confidence*100:.0f}%)"
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                
                # Рисуем прямоугольник около лица и глаза
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color if 'color' in locals() else (0, 0, 255), 2)
    
    cv2.imshow('Iris Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()