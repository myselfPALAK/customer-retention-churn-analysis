"""
Customer Retention & Churn Analysis - Predictive Modeling & Risk Scoring Engine
Trains Logistic Regression & Random Forest classifiers, extracts Odds Ratios and Feature Importances,
and scores individual accounts by Churn Risk.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def prepare_modeling_data(df):
    """
    Encodes categorical features and prepares feature matrix X and target y.
    """
    feature_cols = [
        'subscription_tier',
        'contract_type',
        'billing_cycle',
        'payment_method',
        'paperless_billing',
        'discount_applied',
        'tenure_months',
        'monthly_charges',
        'monthly_active_days',
        'features_adopted',
        'api_calls_monthly',
        'support_tickets',
        'csat_score',
        'days_since_last_active'
    ]
    
    df_model = df[feature_cols].copy()
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    
    X = df_encoded
    y = df['churn']
    
    return X, y, df_encoded.columns.tolist()

def train_and_evaluate_models(X, y, random_state=42):
    """
    Trains Logistic Regression and Random Forest models on 80/20 train/test split.
    
    Returns:
    --------
    results : dict
        Metrics, odds ratios, and feature importances.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    
    # Calculate Odds Ratios
    odds_ratios = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lr.coef_[0],
        'odds_ratio': np.exp(lr.coef_[0])
    }).sort_values(by='odds_ratio', ascending=False)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=random_state)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    # Generate full dataset risk predictions using Random Forest
    full_probs = rf.predict_proba(X)[:, 1]
    
    return {
        'lr_model': lr,
        'rf_model': rf,
        'scaler': scaler,
        'auc_lr': auc_lr,
        'auc_rf': auc_rf,
        'report_lr': classification_report(y_test, y_pred_lr, output_dict=True),
        'report_rf': classification_report(y_test, y_pred_rf, output_dict=True),
        'odds_ratios': odds_ratios,
        'feature_importance': feature_importance,
        'full_risk_probabilities': full_probs
    }

def score_customers(df, model_results):
    """
    Appends churn risk probabilities and risk tier labels to customer dataset.
    """
    df_scored = df.copy()
    df_scored['churn_risk_score'] = np.round(model_results['full_risk_probabilities'], 4)
    
    def assign_risk_tier(score):
        if score >= 0.70:
            return 'Critical (>70%)'
        elif score >= 0.40:
            return 'High (40-70%)'
        elif score >= 0.20:
            return 'Medium (20-40%)'
        else:
            return 'Low (<20%)'
            
    df_scored['risk_tier'] = df_scored['churn_risk_score'].apply(assign_risk_tier)
    return df_scored

if __name__ == '__main__':
    import os
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/customer_churn_data.csv'))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        X, y, feat_names = prepare_modeling_data(df)
        res = train_and_evaluate_models(X, y)
        print(f"Logistic Regression AUC: {res['auc_lr']:.4f}")
        print(f"Random Forest AUC: {res['auc_rf']:.4f}")
        print("\n=== Top 5 Churn Risk Factors (Odds Ratios > 1) ===")
        print(res['odds_ratios'].head(5))
        print("\n=== Top 5 Retention Protective Factors (Odds Ratios < 1) ===")
        print(res['odds_ratios'].tail(5))
        print("\n=== Top 5 Most Important Features (Random Forest) ===")
        print(res['feature_importance'].head(5))
