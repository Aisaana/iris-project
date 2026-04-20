import cv2
import numpy as np
import os
import pickle
from skimage.feature import hog, local_binary_pattern

def extract_features_from_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (60, 36))
    
    # Нормализация освещения
    gray = cv2.equalizeHist(gray)
    
    hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                   visualize=False, feature_vector=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    
    return np.concatenate([hog_feat, lbp_hist])

# Папка с данными
dataset_path = "dataset"
features = []
labels = []

# Проходим по папкам person1, person2, etc.
for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)
    if os.path.isdir(person_path):
        label = int(person_folder.replace("person", "")) - 1  # person1 -> 0, person2 -> 1, etc.

        for image_file in os.listdir(person_path):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(person_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # Поскольку изображения уже ROI глаз, извлекаем признаки напрямую
                feat = extract_features_from_roi(image)
                features.append(feat)
                labels.append(label)

# Сохраняем признаки и метки
with open("models/features.pkl", "wb") as f:
    pickle.dump({"features": np.vstack(features), "labels": np.array(labels)}, f)

print(f"Собрано {len(features)} образцов признаков из {len(set(labels))} человек.")
