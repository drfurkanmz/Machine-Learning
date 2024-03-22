from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Veri setini yükle
dry_bean_dataset = fetch_ucirepo(id=602)

# Özellikler ve hedef değişkeni al
X = dry_bean_dataset.data.features
Y = dry_bean_dataset.data.targets
df1= pd.DataFrame.__array__(Y)
X= pd.DataFrame.__array__(X)
Y_reshaped = df1.ravel()

# Veriyi eğitim ve test kümelerine böle
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_reshaped, train_size=0.7, test_size=0.3, random_state=0, stratify=Y_reshaped)

# MLP sınıflandırıcı modelini oluştur ve eğit
clf = MLPClassifier(random_state=0)
clf.fit(X_train, Y_train)

# Test veri kümesi üzerinde tahmin yap
test_predictions = clf.predict(X_test)

# Doğruluk değerini hesapla ve ekrana yazdır
accuracy = accuracy_score(Y_test, test_predictions)
print("MLP doğruluk değeri:", accuracy)
test = [
    [28395,610.291,208.1781167,173.888747,1.197191425,0.549812187,28715,190.1410973,0.763922518,0.988855999,0.958027126,0.913357755,0.007331506,0.003147289,0.834222388,0.998723889
],
    [27000, 600.456, 190.7890123, 170.6789012, 1.089012345, 0.389012345, 28000, 185.7890123, 0.770123456, 0.970123456, 0.877012345, 0.943012345, 0.005901234, 0.002567890, 0.897012345, 0.998123456]
]

# Tahminleri yap
test_predictions = clf.predict(test)

# Tahmin sonuçlarını ekrana yazdır
print("Test verisi 1 tahmini:", test_predictions[0])
print("Test verisi 2 tahmini:", test_predictions[1])



# Uyarıları filtrele
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 10 katlı çapraz doğrulama ile performans ölçümlerini al
accuracy_scores = cross_val_score(clf, X, Y_reshaped, cv=10, scoring='accuracy')
precision_scores = cross_val_score(clf, X, Y_reshaped, cv=10, scoring='precision_macro')


recall_scores = cross_val_score(clf, X, Y_reshaped, cv=10, scoring='recall_macro')

# Sonuçları bir tabloya kaydetmek için bir sözlük oluştur
results = {
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores
}

# Ortalamaları hesapla
average_results = {k: np.mean(v) for k, v in results.items()}

print("Ortalama sonuçlar:")
for k, v in average_results.items():
    print(f"{k}: {v}")


# Hata matrisini bulmak
y_pred = clf.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)


print("Hata Matrisi:")
print(cm)
