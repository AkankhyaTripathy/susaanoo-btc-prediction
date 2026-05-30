import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from features import FeatureEngineer
from labels import triple_barrier
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'log_return', 'volatility', 'rsi',
    'bb_deviation', 'macd',
    'volume_zscore', 'taker_buy_ratio'
]
def load_test_data():
    df = pd.read_csv("btc_test.csv", index_col="timestamp")
    fe = FeatureEngineer()
    df = fe.compute(df)
    df = triple_barrier(df)
    df = df.dropna()
    return df
def evaluate_models(df):
    lgb_model = joblib.load('lgb_model.pkl')
    lr_model = joblib.load('lr_model.pkl')
    le = joblib.load('label_encoder.pkl')
    
    X = df[FEATURES]
    y = le.transform(df['label'])
    
    lgb_proba = lgb_model.predict_proba(X)
    lr_proba = lr_model.predict_proba(X)
    
    ensemble_proba = (lgb_proba + lr_proba) / 2
    
    prob_long = ensemble_proba[:, 2]
    prob_flat = ensemble_proba[:, 1]
    prob_short = ensemble_proba[:, 0]
    
    predictions = np.argmax(ensemble_proba, axis=1)
    
    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average=None)
    
    non_flat = y != 1
    non_flat_acc = accuracy_score(y[non_flat], predictions[non_flat])
    
    confidence = np.max(ensemble_proba, axis=1)
    
    low_conf = confidence < 0.4
    mid_conf = (confidence >= 0.4) & (confidence < 0.6)
    high_conf = confidence >= 0.6
    
    print("===== EVALUATION RESULTS =====")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Non-flat Directional Accuracy: {non_flat_acc:.4f}")
    print(f"\nPrecision per class: Short={precision[0]:.4f} Flat={precision[1]:.4f} Long={precision[2]:.4f}")
    print(f"Recall per class: Short={recall[0]:.4f} Flat={recall[1]:.4f} Long={recall[2]:.4f}")
    
    print("\nConfidence Stratified Accuracy:")
    if low_conf.sum() > 0:
        print(f"Low confidence: {accuracy_score(y[low_conf], predictions[low_conf]):.4f} ({low_conf.sum()} bars)")
    if mid_conf.sum() > 0:
        print(f"Mid confidence: {accuracy_score(y[mid_conf], predictions[mid_conf]):.4f} ({mid_conf.sum()} bars)")
    if high_conf.sum() > 0:
        print(f"High confidence: {accuracy_score(y[high_conf], predictions[high_conf]):.4f} ({high_conf.sum()} bars)")
    
    print("\nTop 10 Feature Importances (LightGBM):")
    importances = lgb_model.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat}: {imp}")
       
if __name__ == "__main__":
    print("Loading test data...")
    df = load_test_data()
    print(f"Test data shape: {df.shape}")
    evaluate_models(df)