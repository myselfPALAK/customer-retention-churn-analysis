"""
Customer Retention & Churn Analysis - Master End-to-End Execution Pipeline
Runs data generation/ingestion, metrics calculation, multi-dimensional segmentation,
predictive risk modeling, and visualization generation.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import generate_customer_churn_data
from retention_metrics import calculate_kpis, calculate_dimension_churn_rates, calculate_cohort_retention_matrix
from segmentation import apply_rule_based_segmentation, analyze_segment_churn, perform_kmeans_clustering
from modeling import prepare_modeling_data, train_and_evaluate_models, score_customers
from visualizations import (
    plot_cohort_heatmap,
    plot_churn_by_tier_and_contract,
    plot_segment_matrix,
    plot_kmeans_personas,
    plot_feature_importance_and_odds,
    plot_revenue_at_risk
)

def run_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_raw_path = os.path.join(base_dir, 'data/raw/customer_churn_data.csv')
    data_proc_path = os.path.join(base_dir, 'data/processed/customer_segments_scored.csv')
    figures_dir = os.path.join(base_dir, 'reports/figures')
    
    os.makedirs(os.path.dirname(data_proc_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 70)
    print("   CUSTOMER RETENTION & CHURN ANALYSIS - END-TO-END PIPELINE")
    print("=" * 70)
    
    # 1. Load or Generate Data
    if not os.path.exists(data_raw_path):
        print("\n[Step 1/5] Raw data not found. Generating synthetic dataset (n=7,500)...")
        df = generate_customer_churn_data(n_samples=7500, output_path=data_raw_path)
    else:
        print(f"\n[Step 1/5] Loading customer dataset from {data_raw_path}...")
        df = pd.read_csv(data_raw_path)
    print(f"Loaded {len(df):,} customer records across {df['subscription_tier'].nunique()} subscription tiers.")
    
    # 2. Compute Retention KPIs
    print("\n[Step 2/5] Calculating executive KPIs and retention metrics...")
    kpis = calculate_kpis(df)
    print(f" -> Overall Logo Churn Rate: {kpis['logo_churn_rate']:.2%}")
    print(f" -> Overall Retention Rate:  {kpis['retention_rate']:.2%}")
    print(f" -> Total MRR:               ${kpis['total_mrr']:,.2f}")
    print(f" -> Churned MRR:             ${kpis['churned_mrr']:,.2f} ({kpis['mrr_churn_rate']:.2%})")
    print(f" -> Annualized ARR at Risk:  ${kpis['total_revenue_at_risk_arr']:,.2f}")
    
    # 3. Customer Segmentation (Rule-based & K-Means)
    print("\n[Step 3/5] Applying multi-dimensional segmentation & K-Means clustering...")
    df_seg = apply_rule_based_segmentation(df)
    seg_summary = analyze_segment_churn(df_seg)
    
    df_clustered, cluster_profiles, sil_score = perform_kmeans_clustering(df_seg, n_clusters=4)
    print(f" -> K-Means Clustering converged with Silhouette Score: {sil_score:.4f}")
    print(" -> 4 Behavioral Personas identified:")
    for _, row in cluster_profiles.iterrows():
        print(f"    • {row['persona_name']}: {row['account_count']:,} accounts | Churn: {row['churn_rate']:.1%} | Avg MRR: ${row['avg_mrr']:.0f}")
        
    # 4. Predictive Modeling & Churn Risk Scoring
    print("\n[Step 4/5] Training predictive models & scoring account risk levels...")
    X, y, feat_names = prepare_modeling_data(df)
    model_results = train_and_evaluate_models(X, y)
    print(f" -> Logistic Regression ROC-AUC: {model_results['auc_lr']:.4f}")
    print(f" -> Random Forest ROC-AUC:       {model_results['auc_rf']:.4f}")
    
    df_final = score_customers(df_clustered, model_results)
    df_final.to_csv(data_proc_path, index=False)
    print(f" -> Saved processed & scored customer data to: {data_proc_path}")
    
    # 5. Visualizations Export
    print("\n[Step 5/5] Generating publication-grade figures for reports & README...")
    cohort_df = calculate_cohort_retention_matrix(df)
    plot_cohort_heatmap(cohort_df, os.path.join(figures_dir, 'cohort_retention_heatmap.png'))
    plot_churn_by_tier_and_contract(df, os.path.join(figures_dir, 'churn_by_tier_and_contract.png'))
    plot_segment_matrix(df_final, os.path.join(figures_dir, 'customer_segments_matrix.png'))
    plot_kmeans_personas(cluster_profiles, os.path.join(figures_dir, 'kmeans_cluster_profiles.png'))
    plot_feature_importance_and_odds(model_results, os.path.join(figures_dir, 'churn_risk_feature_importance.png'))
    plot_revenue_at_risk(df, os.path.join(figures_dir, 'revenue_at_risk_waterfall.png'))
    
    print("\n" + "=" * 70)
    print("   PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS SUCCESSFULLY GENERATED")
    print("=" * 70)
    
    return {
        'kpis': kpis,
        'seg_summary': seg_summary,
        'cluster_profiles': cluster_profiles,
        'model_results': model_results,
        'df_final': df_final
    }

if __name__ == '__main__':
    run_pipeline()
