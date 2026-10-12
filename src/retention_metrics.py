"""
Customer Retention & Churn Analysis - Retention Metrics Engine
Calculates Logo Churn, MRR Churn, Cohort Retention Matrices, and Customer Lifetime Value (LTV).
"""

import numpy as np
import pandas as pd

def calculate_kpis(df):
    """
    Calculate top-line executive KPIs for customer base and revenue.
    """
    total_customers = len(df)
    churned_customers = df['churn'].sum()
    retained_customers = total_customers - churned_customers
    logo_churn_rate = churned_customers / total_customers
    retention_rate = 1.0 - logo_churn_rate
    
    total_mrr = df['monthly_charges'].sum()
    churned_mrr = df[df['churn'] == 1]['monthly_charges'].sum()
    retained_mrr = df[df['churn'] == 0]['monthly_charges'].sum()
    mrr_churn_rate = churned_mrr / total_mrr
    
    arpu = df['monthly_charges'].mean()
    # Estimated Lifetime in months = 1 / monthly_churn_rate (approx annualized / monthly)
    # Using logo churn rate across period
    avg_tenure = df['tenure_months'].mean()
    est_ltv = arpu * (avg_tenure / (logo_churn_rate + 0.001)) # heuristic LTV
    
    return {
        'total_customers': int(total_customers),
        'retained_customers': int(retained_customers),
        'churned_customers': int(churned_customers),
        'logo_churn_rate': float(logo_churn_rate),
        'retention_rate': float(retention_rate),
        'total_mrr': float(total_mrr),
        'churned_mrr': float(churned_mrr),
        'retained_mrr': float(retained_mrr),
        'mrr_churn_rate': float(mrr_churn_rate),
        'arpu': float(arpu),
        'avg_tenure_months': float(avg_tenure),
        'total_revenue_at_risk_arr': float(churned_mrr * 12)
    }

def calculate_dimension_churn_rates(df, dimension_col):
    """
    Calculates detailed churn and retention rates broken down by a specific categorical dimension.
    """
    summary = df.groupby(dimension_col).agg(
        total_accounts=('customer_id', 'count'),
        churned_accounts=('churn', 'sum'),
        total_mrr=('monthly_charges', 'sum'),
        churned_mrr=('monthly_charges', lambda x: x[df.loc[x.index, 'churn'] == 1].sum()),
        avg_active_days=('monthly_active_days', 'mean'),
        avg_features=('features_adopted', 'mean'),
        avg_csat=('csat_score', 'mean')
    ).reset_index()
    
    summary['retained_accounts'] = summary['total_accounts'] - summary['churned_accounts']
    summary['churn_rate'] = (summary['churned_accounts'] / summary['total_accounts']).round(4)
    summary['retention_rate'] = (1 - summary['churn_rate']).round(4)
    summary['mrr_churn_rate'] = (summary['churned_mrr'] / summary['total_mrr']).round(4)
    summary['mrr_at_risk_arr'] = (summary['churned_mrr'] * 12).round(2)
    
    return summary.sort_values(by='churn_rate', ascending=False)

def calculate_cohort_retention_matrix(df, max_periods=12):
    """
    Constructs a monthly signup cohort retention triangle.
    Tracks percentage of customers remaining active across tenure months 0 to max_periods.
    """
    temp_df = df.copy()
    temp_df['signup_cohort'] = pd.to_datetime(temp_df['signup_date']).dt.to_period('M')
    
    cohorts = temp_df['signup_cohort'].unique()
    cohorts = sorted(cohorts)
    
    cohort_data = []
    
    for cohort in cohorts:
        cohort_users = temp_df[temp_df['signup_cohort'] == cohort]
        cohort_size = len(cohort_users)
        
        row = {'cohort': str(cohort), 'cohort_size': cohort_size}
        
        for period in range(max_periods + 1):
            if period == 0:
                row[f'M{period}'] = 1.0
            else:
                # Active if tenure >= period and (not churned OR churn occurred at/after period)
                # For synthetic data, customers with tenure_months >= period and not churned before period
                # If churned, they churned at their tenure_months mark
                retained_count = cohort_users[
                    (cohort_users['tenure_months'] >= period) & 
                    ((cohort_users['churn'] == 0) | (cohort_users['tenure_months'] > period))
                ].shape[0]
                
                # Check if cohort has matured up to this period
                # Observation window ends 2025-12
                months_elapsed = (pd.Period('2025-12', freq='M') - cohort).n
                if months_elapsed >= period:
                    row[f'M{period}'] = round(retained_count / cohort_size, 3)
                else:
                    row[f'M{period}'] = np.nan
                    
        cohort_data.append(row)
        
    cohort_df = pd.DataFrame(cohort_data).set_index('cohort')
    return cohort_df

def calculate_tenure_survival_curve(df):
    """
    Calculates overall empirical survival probability curve by tenure month.
    """
    survival = []
    total = len(df)
    for m in range(1, 25):
        # Count retained at tenure >= m
        active_at_m = df[(df['tenure_months'] >= m) & ((df['churn'] == 0) | (df['tenure_months'] > m))].shape[0]
        eligible = df[df['tenure_months'] >= m].shape[0]
        rate = active_at_m / eligible if eligible > 0 else 0
        survival.append({'tenure_month': m, 'retention_rate': rate, 'eligible_customers': eligible})
    return pd.DataFrame(survival)

if __name__ == '__main__':
    import os
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/customer_churn_data.csv'))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        kpis = calculate_kpis(df)
        print("=== Topline KPIs ===")
        for k, v in kpis.items():
            print(f"{k}: {v}")
        
        print("\n=== Churn by Subscription Tier ===")
        print(calculate_dimension_churn_rates(df, 'subscription_tier')[['subscription_tier', 'total_accounts', 'churn_rate', 'mrr_churn_rate']])
        
        print("\n=== Churn by Contract Type ===")
        print(calculate_dimension_churn_rates(df, 'contract_type')[['contract_type', 'total_accounts', 'churn_rate', 'mrr_churn_rate']])
