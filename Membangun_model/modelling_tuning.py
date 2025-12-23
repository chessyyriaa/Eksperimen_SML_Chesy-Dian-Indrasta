import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import dagshub.auth
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import shutil

# --- KONFIGURASI ---
USER_DAGSHUB = "chessyyriaa"
REPO_NAME = "Eksperimen_SML_Chesy-Dian-Indrasta"
MY_TOKEN = os.environ.get("DAGSHUB_USER_TOKEN") 


def clear_dagshub_cache():
    try:
        if "DAGSHUB_USER_TOKEN" in os.environ:
            del os.environ["DAGSHUB_USER_TOKEN"]
        dagshub.auth.add_app_token(MY_TOKEN)
    except Exception as e:
        print(f"Peringatan saat clear cache: {e}")

def load_data(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {full_path}")
    
    print(f"Membaca data dari: {full_path}")
    return pd.read_csv(full_path)

def main():
    # 1. Bersihkan Cache Lama & Setup Auth
    clear_dagshub_cache()
    
    # 2. Setup Environment Variables untuk MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = USER_DAGSHUB
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MY_TOKEN
    
    print(f"Menghubungkan ke DagsHub repo: {REPO_NAME}...")
    dagshub.init(repo_owner=USER_DAGSHUB, repo_name=REPO_NAME, mlflow=True)
    
    # 3. Setup Eksperimen (Dengan Error Handling)
    experiment_name = "Eksperimen_Churn_Prediction"
    try:
        mlflow.set_experiment(experiment_name)
    except:
        print(f"Eksperimen '{experiment_name}' belum ada. Membuat baru...")
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Gagal membuat eksperimen: {e}")

    # 4. Load Data (Menggunakan fungsi path baru)
    print("Loading data...")
    df = load_data("churn_data_clean.csv")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Hyperparameter Tuning Setup
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10], # Dikurangi biar cepat
        'min_samples_split': [2, 5]
    }

    # Mulai Tracking MLflow
    with mlflow.start_run(run_name="Hyperparameter_Tuning_RandomForest"):
        print("🚀 Memulai Training... (Tunggu sebentar)")
        
        # Grid Search
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"✅ Best Params: {best_params}")

        # Prediksi
        y_pred = best_model.predict(X_test)

        # Hitung Metriks
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"📊 Accuracy: {accuracy}")

        # --- LOGGING KE MLFLOW ---
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        mlflow.sklearn.log_model(best_model, "model")

        # Log Artefak (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        
        # Simpan gambar di folder script berada
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(base_dir, "confusion_matrix.png")
        
        plt.savefig(image_path)
        mlflow.log_artifact(image_path)
        
        # Hapus file gambar lokal
        if os.path.exists(image_path):
            os.remove(image_path)
        
        print("🎉 Selesai! Cek DagsHub kamu sekarang.")

if __name__ == "__main__":

    main()
