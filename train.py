import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')
from features import FeatureEngineer
from labels import triple_barrier

FEATURES = [
    'log_return', 'volatility', 'rsi', 
    'bb_deviation', 'macd', 
    'volume_zscore', 'taker_buy_ratio'
]
def load_data():
    df = pd.read_csv("btc_data.csv", index_col="timestamp")
    fe = FeatureEngineer()
    df = fe.compute(df)
    df = triple_barrier(df)
    df = df.dropna()
    return df
def get_folds(df, n_folds=5, gap=24):
    fold_size = len(df) // n_folds
    folds = []
    
    for i in range(n_folds - 1):
        train_end = (i + 1) * fold_size
        val_start = train_end + gap
        val_end = val_start + fold_size
        
        if val_end > len(df):
            break
            
        train_idx = list(range(0, train_end))
        val_idx = list(range(val_start, val_end))
        folds.append((train_idx, val_idx))
    
    return folds
def train_models(df):
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    X = df[FEATURES]
    
    folds = get_folds(df)
    
    lgb_scores = []
    lr_scores = []
    
    for i, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_val)
        lgb_acc = accuracy_score(y_val, lgb_pred)
        lgb_scores.append(lgb_acc)
        
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_val)
        lr_acc = accuracy_score(y_val, lr_pred)
        lr_scores.append(lr_acc)
        
        print(f"Fold {i+1} — LightGBM: {lgb_acc:.4f} | LogReg: {lr_acc:.4f}")
    
    print(f"\nAverage LightGBM accuracy: {np.mean(lgb_scores):.4f}")
    print(f"Average LogReg accuracy: {np.mean(lr_scores):.4f}")
    
    lgb_final = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    lgb_final.fit(X, y)
    
    lr_final = LogisticRegression(max_iter=1000, random_state=42)
    lr_final.fit(X, y)
    
    joblib.dump(lgb_final, 'lgb_model.pkl')
    joblib.dump(lr_final, 'lr_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    print("\nModels saved!")
    return lgb_final, lr_final, le

def position_size(prob_long, prob_short, prob_flat, confidence):
    dominant = max(prob_long, prob_short, prob_flat)
    
    if dominant == prob_flat:
        return 0
    
    if abs(prob_long - prob_short) < 0.1:
        return 0
    
    if prob_long > prob_short:
        direction = 1
    else:
        direction = -1
    
    size = direction * confidence
    size = max(-1, min(1, size))
    return size

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Data shape: {df.shape}")
    
    majority = df['label'].value_counts(normalize=True).max()
    print(f"Majority class baseline: {majority:.4f}")
    
    print("\nTraining models...")
    train_models(df)

    print("\nTesting position sizing:")
    print(position_size(0.6, 0.2, 0.2, 0.8))
    print(position_size(0.2, 0.6, 0.2, 0.8))
    print(position_size(0.1, 0.1, 0.8, 0.9))
    print(position_size(0.4, 0.35, 0.25, 0.5))
    
