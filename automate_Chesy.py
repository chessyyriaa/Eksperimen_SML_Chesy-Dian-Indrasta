import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_data(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"CRITICAL ERROR: File tidak ditemukan di: {path}")
    print(f"Membaca data dari {path}...")
    return pd.read_csv(path)

def preprocess_data(df):
    print("Memulai preprocessing...")
    
    # Cleaning TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Encoding
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        
    print("Preprocessing selesai.")
    return df

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data bersih disimpan di: {output_path}")

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(base_dir, 'data_raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    output_file = os.path.join(base_dir, 'preprocessing', 'churn_data_clean.csv')
    
    try:
        df = load_data(input_file)
        df_clean = preprocess_data(df)
        save_data(df_clean, output_file)
        print("Done! Sukses preprocessing.")
    except Exception as e:
        print(f"TERJADI ERROR: {e}")
        exit(1) 