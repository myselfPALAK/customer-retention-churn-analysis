"""
Customer Retention & Churn Analysis - Synthetic Data Generator
Generates realistic, reproducible B2B SaaS customer behavior and subscription dataset.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_customer_churn_data(n_samples=7500, random_state=42, output_path=None):
    """
    Generate realistic customer records with grounded behavioral and contract features.
    
    Parameters:
    -----------
    n_samples : int
        Number of customer records to generate (default 7500).
    random_state : int
        Random seed for reproducibility.
    output_path : str, optional
        Filepath to save the generated CSV.
        
    Returns:
    --------
    pd.DataFrame
        Synthetic customer dataset.
    """
    np.random.seed(random_state)
    
    # Customer IDs
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    
    # Observation reference date: Dec 31, 2025
    ref_date = datetime(2025, 12, 31)
    
    # Signup Dates distributed between Jan 2024 and Oct 2025 (22 months)
    start_date = datetime(2024, 1, 1)
    date_range_days = (datetime(2025, 10, 31) - start_date).days
    random_days = np.random.randint(0, date_range_days, size=n_samples)
    signup_dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # Subscription Tiers & Base Pricing
    # 45% Starter, 40% Professional, 15% Enterprise
    tier_choices = ['Starter', 'Professional', 'Enterprise']
    tier_probs = [0.45, 0.40, 0.15]
    subscription_tiers = np.random.choice(tier_choices, size=n_samples, p=tier_probs)
    
    base_prices = {'Starter': 29.0, 'Professional': 99.0, 'Enterprise': 299.0}
    
    # Contract Types
    # Month-to-Month (55%), One-Year (32%), Two-Year (13%)
    contract_choices = ['Month-to-Month', 'One-Year', 'Two-Year']
    contract_probs = [0.55, 0.32, 0.13]
    contract_types = np.random.choice(contract_choices, size=n_samples, p=contract_probs)
    
    # Billing Cycle
    billing_cycles = []
    for c in contract_types:
        if c == 'Month-to-Month':
            billing_cycles.append('Monthly')
        else:
            billing_cycles.append(np.random.choice(['Monthly', 'Annual'], p=[0.25, 0.75]))
            
    # Payment Methods
    payment_choices = ['Credit Card', 'Bank Transfer (ACH)', 'PayPal']
    payment_probs = [0.58, 0.28, 0.14]
    payment_methods = np.random.choice(payment_choices, size=n_samples, p=payment_probs)
    
    # Paperless Billing
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.78, 0.22])
    
    # Discounts (Promotions)
    discounts = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.20, 0.80])
    
    # Tenure in months (up to reference date)
    tenure_months = []
    for s_date in signup_dates:
        diff_months = max(1, (ref_date.year - s_date.year) * 12 + (ref_date.month - s_date.month))
        # Cap tenure at 24 months
        tenure_months.append(min(24, diff_months))
    tenure_months = np.array(tenure_months)
    
    # Monthly charges with slight seat/usage variance
    monthly_charges = []
    for tier, disc in zip(subscription_tiers, discounts):
        base = base_prices[tier]
        seat_variance = np.random.uniform(0.9, 1.25)
        price = base * seat_variance
        if disc == 'Yes':
            price *= 0.85 # 15% promotional discount
        monthly_charges.append(round(price, 2))
    monthly_charges = np.array(monthly_charges)
    
    # Total charges
    total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.96, 1.02, size=n_samples), 2)
    
    # Behavioral features
    # 1. Monthly active days (0 to 30)
    # Higher for Enterprise & Pro, correlated with tenure
    monthly_active_days = []
    for tier in subscription_tiers:
        if tier == 'Enterprise':
            days = np.clip(np.random.normal(22, 5), 1, 30)
        elif tier == 'Professional':
            days = np.clip(np.random.normal(16, 6), 1, 30)
        else: # Starter
            days = np.clip(np.random.normal(11, 7), 1, 30)
        monthly_active_days.append(int(days))
    monthly_active_days = np.array(monthly_active_days)
    
    # 2. Features adopted (1 to 8 features)
    features_adopted = []
    for tier, active in zip(subscription_tiers, monthly_active_days):
        tier_max = 8 if tier == 'Enterprise' else (6 if tier == 'Professional' else 4)
        active_ratio = active / 30.0
        feat = int(np.clip(np.random.poisson(lam=tier_max * active_ratio + 1), 1, tier_max))
        features_adopted.append(feat)
    features_adopted = np.array(features_adopted)
    
    # 3. Monthly API Calls
    api_calls = []
    for tier, active in zip(subscription_tiers, monthly_active_days):
        if tier == 'Enterprise':
            base_api = np.random.exponential(12000) + 5000
        elif tier == 'Professional':
            base_api = np.random.exponential(3500) + 800
        else:
            base_api = np.random.exponential(600) + 50
        api_calls.append(int(base_api * (active / 20.0)))
    api_calls = np.array(api_calls)
    
    # 4. Support Tickets (0 to 10)
    support_tickets = np.random.poisson(lam=1.8, size=n_samples)
    support_tickets = np.clip(support_tickets, 0, 10)
    
    # 5. CSAT Score (1 to 5)
    # Negatively correlated with support tickets, positively with active days
    csat_scores = []
    for tickets, active in zip(support_tickets, monthly_active_days):
        base_csat = 4.2 - (tickets * 0.4) + (active * 0.04)
        score = int(np.clip(np.round(base_csat + np.random.normal(0, 0.7)), 1, 5))
        csat_scores.append(score)
    csat_scores = np.array(csat_scores)
    
    # 6. Days Since Last Active (recency)
    days_since_last_active = []
    for active in monthly_active_days:
        if active > 20:
            recency = np.random.randint(0, 4)
        elif active > 10:
            recency = np.random.randint(2, 14)
        else:
            recency = np.random.randint(7, 45)
        days_since_last_active.append(recency)
    days_since_last_active = np.array(days_since_last_active)
    
    # Ground-truth Churn Probability Engine
    # Logistic sigmoid based on domain-tested risk factors
    log_odds = -2.2 # Base baseline log-odds (~10% baseline)
    
    # Contract Type risk
    contract_risk = np.where(contract_types == 'Month-to-Month', 1.35, 
                    np.where(contract_types == 'One-Year', -0.65, -1.25))
    
    # Early tenure friction (Month 1-3 onboarding cliff)
    tenure_risk = np.where(tenure_months <= 3, 1.20,
                  np.where(tenure_months <= 6, 0.40,
                  np.where(tenure_months > 12, -0.75, 0.0)))
                  
    # Inactivity / Usage risk
    activity_risk = (15 - monthly_active_days) * 0.11
    
    # Low feature adoption risk
    feature_risk = (4 - features_adopted) * 0.28
    
    # Support & CSAT friction
    support_risk = (support_tickets - 2) * 0.22
    csat_risk = (3.5 - csat_scores) * 0.48
    
    # Recency dormancy
    recency_risk = np.where(days_since_last_active > 20, 1.10, 
                   np.where(days_since_last_active > 10, 0.45, -0.30))
                   
    # Tier risk (Starters churn more, Enterprises churn less)
    tier_risk = np.where(subscription_tiers == 'Starter', 0.50,
                np.where(subscription_tiers == 'Professional', -0.15, -0.70))
                
    # Total logit
    total_logit = (log_odds + contract_risk + tenure_risk + activity_risk + 
                   feature_risk + support_risk + csat_risk + recency_risk + tier_risk)
                   
    churn_probs = 1.0 / (1.0 + np.exp(-total_logit))
    
    # Generate binary churn outcome
    churn = (np.random.uniform(0, 1, size=n_samples) < churn_probs).astype(int)
    
    # Determine churn date & reason
    churn_reasons = [
        'High Price / Budget Cuts',
        'Competitor Switch',
        'Poor Support Experience',
        'Low Product Adoption',
        'Missing Core Features'
    ]
    
    churn_dates = []
    assigned_reasons = []
    
    for i in range(n_samples):
        if churn[i] == 1:
            # Churn date happens within their tenure
            signup_d = signup_dates[i]
            active_duration_days = min((ref_date - signup_d).days, int(tenure_months[i] * 30.4))
            churn_d = signup_d + timedelta(days=max(15, active_duration_days - np.random.randint(1, 15)))
            if churn_d > ref_date:
                churn_d = ref_date - timedelta(days=np.random.randint(1, 20))
            churn_dates.append(churn_d.strftime('%Y-%m-%d'))
            
            # Context-sensitive churn reason
            if csat_scores[i] <= 2 or support_tickets[i] >= 4:
                reason = 'Poor Support Experience'
            elif features_adopted[i] <= 2 or monthly_active_days[i] <= 5:
                reason = 'Low Product Adoption'
            elif monthly_charges[i] >= 99 and np.random.rand() > 0.45:
                reason = 'High Price / Budget Cuts'
            elif np.random.rand() > 0.5:
                reason = 'Competitor Switch'
            else:
                reason = 'Missing Core Features'
            assigned_reasons.append(reason)
        else:
            churn_dates.append(None)
            assigned_reasons.append(None)
            
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'signup_date': [d.strftime('%Y-%m-%d') for d in signup_dates],
        'subscription_tier': subscription_tiers,
        'contract_type': contract_types,
        'billing_cycle': billing_cycles,
        'payment_method': payment_methods,
        'paperless_billing': paperless_billing,
        'discount_applied': discounts,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'monthly_active_days': monthly_active_days,
        'features_adopted': features_adopted,
        'api_calls_monthly': api_calls,
        'support_tickets': support_tickets,
        'csat_score': csat_scores,
        'days_since_last_active': days_since_last_active,
        'churn': churn,
        'churn_date': churn_dates,
        'churn_reason': assigned_reasons
    })
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully generated {len(df)} records and saved to {output_path}")
        print(f"Overall Churn Rate: {df['churn'].mean():.2%}")
        print(f"Tiers: {df['subscription_tier'].value_counts().to_dict()}")
        
    return df

if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw'))
    csv_file = os.path.join(data_dir, 'customer_churn_data.csv')
    generate_customer_churn_data(n_samples=7500, output_path=csv_file)
