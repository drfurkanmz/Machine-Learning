from ucimlrepo import fetch_ucirepo 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pandas as pd  
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from sklearn.tree import DecisionTreeClassifier
#  dataseti al
dry_bean_dataset = fetch_ucirepo(id=602) 

X = dry_bean_dataset.data.features #değerleri aldim
Y = dry_bean_dataset.data.targets #sınıf değerlerini alır 

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X,Y)

test = [
    [28395,610.291,208.1781167,173.888747,1.197191425,0.549812187,28715,190.1410973,0.763922518,0.988855999,0.958027126,0.913357755,0.007331506,0.003147289,0.834222388,0.998723889
],
    [27000, 600.456, 190.7890123, 170.6789012, 1.089012345, 0.389012345, 28000, 185.7890123, 0.770123456, 0.970123456, 0.877012345, 0.943012345, 0.005901234, 0.002567890, 0.897012345, 0.998123456]
]

test_sonuc = clf.predict(test)
print(test_sonuc)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.8, test_size =0.2, random_state = 0, stratify = Y)
clf.fit(X_train,Y_train)

test_sonuc = clf.predict(X_test)
#print(test_sonuc)
print('Karar ağaci doğruluk değeri: ' + str(accuracy_score(test_sonuc, Y_test)))

accuracy_scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')
precision_scores = cross_val_score(clf, X, Y, cv=10, scoring='precision_macro')
recall_scores = cross_val_score(clf, X, Y, cv=10, scoring='recall_macro')

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

