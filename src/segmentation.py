"""
Customer Retention & Churn Analysis - Customer Segmentation Engine
Performs multi-dimensional rule-based segmentation (Tenure, Usage, Subscription Tier)
and unsupervised behavioral clustering (K-Means).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def apply_rule_based_segmentation(df):
    """
    Segment customers across three core business dimensions:
    1. Tenure Stage
    2. Usage Intensity
    3. Subscription Tier
    """
    df_seg = df.copy()
    
    # 1. Tenure Stage
    def categorize_tenure(tenure):
        if tenure <= 3:
            return '1. Onboarding (1-3m)'
        elif tenure <= 12:
            return '2. Growth (4-12m)'
        else:
            return '3. Mature (13-24m)'
            
    df_seg['tenure_segment'] = df_seg['tenure_months'].apply(categorize_tenure)
    
    # 2. Usage Intensity
    def categorize_usage(row):
        active = row['monthly_active_days']
        features = row['features_adopted']
        
        if active <= 8 or features <= 2:
            return 'Low / Dormant'
        elif active >= 19 and features >= 5:
            return 'Power User'
        else:
            return 'Moderate User'
            
    df_seg['usage_segment'] = df_seg.apply(categorize_usage, axis=1)
    
    # 3. Composite Segment Code
    df_seg['composite_segment'] = (
        df_seg['subscription_tier'] + " | " + 
        df_seg['tenure_segment'].str.split('. ').str[-1] + " | " + 
        df_seg['usage_segment']
    )
    
    return df_seg

def analyze_segment_churn(df_seg):
    """
    Calculates detailed retention and churn metrics for each segment.
    """
    summary = df_seg.groupby('composite_segment').agg(
        accounts=('customer_id', 'count'),
        churned=('churn', 'sum'),
        total_mrr=('monthly_charges', 'sum'),
        churned_mrr=('monthly_charges', lambda x: x[df_seg.loc[x.index, 'churn'] == 1].sum()),
        avg_tenure=('tenure_months', 'mean'),
        avg_active_days=('monthly_active_days', 'mean'),
        avg_features=('features_adopted', 'mean'),
        avg_tickets=('support_tickets', 'mean'),
        avg_csat=('csat_score', 'mean')
    ).reset_index()
    
    summary['churn_rate'] = (summary['churned'] / summary['accounts']).round(4)
    summary['retention_rate'] = (1.0 - summary['churn_rate']).round(4)
    summary['mrr_churn_rate'] = (summary['churned_mrr'] / summary['total_mrr']).round(4)
    summary['annual_revenue_at_risk'] = (summary['churned_mrr'] * 12).round(2)
    summary['pct_of_total_churn'] = (summary['churned'] / df_seg['churn'].sum()).round(4)
    
    # High churn flag: churn_rate > 35%
    summary['risk_level'] = np.where(summary['churn_rate'] >= 0.40, 'Critical (>40%)',
                             np.where(summary['churn_rate'] >= 0.25, 'Elevated (25-40%)', 'Low (<25%)'))
                             
    return summary.sort_values(by='churn_rate', ascending=False)

def perform_kmeans_clustering(df, n_clusters=4, random_state=42):
    """
    Performs unsupervised behavioral clustering using normalized engagement features.
    
    Returns:
    --------
    df_clustered : pd.DataFrame
        DataFrame with cluster assignments and human-readable persona names.
    cluster_profiles : pd.DataFrame
        Mean metrics per cluster.
    silhouette : float
        Overall silhouette score.
    """
    feature_cols = [
        'tenure_months',
        'monthly_charges',
        'monthly_active_days',
        'features_adopted',
        'api_calls_monthly',
        'support_tickets',
        'csat_score',
        'days_since_last_active'
    ]
    
    X = df[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, cluster_labels)
    
    df_clustered = df.copy()
    df_clustered['cluster_id'] = cluster_labels
    
    # Generate profiles
    profile = df_clustered.groupby('cluster_id').agg(
        account_count=('customer_id', 'count'),
        churn_rate=('churn', 'mean'),
        mrr_total=('monthly_charges', 'sum'),
        mrr_churned=('monthly_charges', lambda x: x[df_clustered.loc[x.index, 'churn'] == 1].sum()),
        avg_mrr=('monthly_charges', 'mean'),
        avg_tenure=('tenure_months', 'mean'),
        avg_active_days=('monthly_active_days', 'mean'),
        avg_features=('features_adopted', 'mean'),
        avg_api_calls=('api_calls_monthly', 'mean'),
        avg_tickets=('support_tickets', 'mean'),
        avg_csat=('csat_score', 'mean'),
        avg_recency_days=('days_since_last_active', 'mean')
    ).reset_index()
    
    # Map clusters to descriptive persona names dynamically based on characteristics
    persona_mapping = {}
    for idx, row in profile.iterrows():
        c_id = row['cluster_id']
        if row['avg_mrr'] > 200:
            name = "Enterprise Powerhouses"
        elif row['avg_active_days'] < 10 or row['churn_rate'] > 0.50:
            name = "Dormant Flight Risks"
        elif row['avg_tickets'] > 2.5 or row['avg_csat'] < 3.5:
            name = "Support Friction & Dissatisfied"
        else:
            name = "Satisfied Core Adopters"
        persona_mapping[c_id] = name
        
    df_clustered['persona_name'] = df_clustered['cluster_id'].map(persona_mapping)
    profile['persona_name'] = profile['cluster_id'].map(persona_mapping)
    profile['churn_rate'] = profile['churn_rate'].round(4)
    profile['annual_revenue_at_risk'] = (profile['mrr_churned'] * 12).round(2)
    
    return df_clustered, profile, sil_score

if __name__ == '__main__':
    import os
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/customer_churn_data.csv'))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df_seg = apply_rule_based_segmentation(df)
        seg_summary = analyze_segment_churn(df_seg)
        print("=== Top 5 Highest Churn Segments (Rule-Based) ===")
        print(seg_summary[['composite_segment', 'accounts', 'churn_rate', 'risk_level']].head(5))
        
        df_clustered, profiles, sil = perform_kmeans_clustering(df_seg)
        print(f"\nK-Means Silhouette Score: {sil:.4f}")
        print("=== K-Means Behavioral Personas ===")
        print(profiles[['persona_name', 'account_count', 'churn_rate', 'avg_mrr', 'avg_active_days', 'avg_csat']])
