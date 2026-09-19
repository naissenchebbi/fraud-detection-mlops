import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Charger les données
print("📥 Chargement des données...")
df = pd.read_csv('creditcard.csv')
print(f"✅ Dataset chargé : {len(df)} transactions")

# 2. Séparer features et target
X = df.drop('Class', axis=1)
y = df['Class']

# 3. Split AVANT SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scaling (essentiel pour LR et KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. SMOTE sur le train
print("\n⚖️ Application de SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"Après SMOTE : {len(X_train_res)} échantillons")


print("\n" + "="*50)
print("📊 LOGISTIC REGRESSION")
print("="*50)

lr_model=LogisticRegression(
    max_iter=1000,class_weight='balanced',random_state=42)

lr_model.fit(X_train_res,y_train_res)

y_pred_lr=lr_model.predict(X_test_scaled)

y_pred_lr=lr_model.predict(X_test_scaled)
y_proba_lr=lr_model.predict_proba(X_test_scaled)[:,1]

print(classification_report(y_test,y_pred_lr))
print(f"AUC-ROC : {roc_auc_score(y_test,y_proba_lr):.4f}")
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred_lr))

#KNN
print("\n" + "="*50)
print("📊 KNN (K=5)")
print("="*50)

knn_model=KNeighborsClassifier(
    n_neighbors=5,n_jobs=-1)

knn_model.fit(X_train_res, y_train_res)

y_pred_knn=knn_model.predict(X_test_scaled)
y_proba_knn=knn_model.predict_proba(X_test_scaled)[:,1]

print(classification_report(y_test, y_pred_knn))
print(f"AUC-ROC : {roc_auc_score(y_test, y_proba_knn):.4f}")
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred_knn))


joblib.dump(lr_model,'fraud_model_lr.pkl')
joblib.dump(knn_model,'fraud_model_knn.pkl')

print("\n✅ Modèles sauvegardés !")

