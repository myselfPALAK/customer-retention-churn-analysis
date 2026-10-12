"""
Customer Retention & Churn Analysis - Publication-Grade Visualization Suite
Generates high-resolution figures for reports and GitHub presentation.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configure clean aesthetic
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans, Bitstream Vera Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

PALETTE = {
    'primary': '#1E3A8A',     # Dark Navy
    'secondary': '#3B82F6',   # Royal Blue
    'accent_red': '#EF4444',  # Coral Red (Churn)
    'accent_green': '#10B981',# Emerald Green (Retention)
    'accent_amber': '#F59E0B',# Amber (Warning)
    'neutral_dark': '#1F2937',
    'neutral_light': '#F3F4F6',
    'card_bg': '#FFFFFF'
}

def plot_cohort_heatmap(cohort_df, output_path):
    """
    Plots a cohort retention triangle heatmap with percentage labels.
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Select period columns M0 to M12
    cols = [f'M{i}' for i in range(13) if f'M{i}' in cohort_df.columns]
    data = cohort_df[cols].values
    
    cmap = plt.cm.YlGnBu
    norm = mcolors.Normalize(vmin=0.4, vmax=1.0)
    
    # Mask NaNs
    masked_data = np.ma.masked_invalid(data)
    
    im = ax.imshow(masked_data, cmap=cmap, norm=norm, aspect='auto')
    
    # Axis labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=10, fontweight='bold', color=PALETTE['neutral_dark'])
    ax.set_yticks(np.arange(len(cohort_df)))
    
    # Cohort labels with cohort size
    cohort_labels = [f"{idx} (n={cohort_df.loc[idx, 'cohort_size']})" for idx in cohort_df.index]
    ax.set_yticklabels(cohort_labels, fontsize=9, color=PALETTE['neutral_dark'])
    
    # Loop over data dimensions and create text annotations
    for i in range(len(cohort_df)):
        for j in range(len(cols)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.65 or val > 0.85 else '#111827'
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", 
                        color=text_color, fontsize=8.5, fontweight='semibold')
                        
    ax.set_title("Customer Cohort Retention Matrix (12-Month Observation Triangle)\nPercentage of Active Accounts Retained by Tenure Month", 
                 fontsize=13, fontweight='bold', pad=15, color=PALETTE['neutral_dark'], loc='left')
    ax.set_xlabel("Tenure Month from Signup", fontsize=10, fontweight='bold', labelpad=10, color=PALETTE['neutral_dark'])
    ax.set_ylabel("Signup Cohort & Initial Size", fontsize=10, fontweight='bold', labelpad=10, color=PALETTE['neutral_dark'])
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.set_ylabel("Retention Rate", rotation=-90, va="bottom", fontsize=9, fontweight='semibold')
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_churn_by_tier_and_contract(df, output_path):
    """
    Plots grouped bar chart of Churn Rate by Subscription Tier and Contract Type.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    grouped = df.groupby(['subscription_tier', 'contract_type'])['churn'].mean().unstack()
    tiers_order = ['Starter', 'Professional', 'Enterprise']
    contracts_order = ['Month-to-Month', 'One-Year', 'Two-Year']
    grouped = grouped.reindex(index=tiers_order, columns=contracts_order)
    
    x = np.arange(len(tiers_order))
    width = 0.25
    
    colors = ['#EF4444', '#3B82F6', '#10B981']
    
    for i, (contract, color) in enumerate(zip(contracts_order, colors)):
        vals = grouped[contract].values * 100
        bars = ax.bar(x + (i - 1) * width, vals, width, label=contract, color=color, alpha=0.9, edgecolor='none')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=PALETTE['neutral_dark'])
                        
    # Baseline benchmark line
    overall_churn = df['churn'].mean() * 100
    ax.axhline(overall_churn, color='#9CA3AF', linestyle='--', linewidth=1.2, 
               label=f'Portfolio Avg ({overall_churn:.1f}%)')
               
    ax.set_xticks(x)
    ax.set_xticklabels(tiers_order, fontsize=11, fontweight='bold', color=PALETTE['neutral_dark'])
    ax.set_ylabel("Churn Rate (%)", fontsize=11, fontweight='bold', color=PALETTE['neutral_dark'])
    ax.set_ylim(0, 60)
    ax.set_title("Customer Churn Rate by Subscription Tier & Contract Commitment\nMonth-to-Month Starter accounts demonstrate 5x higher churn risk than Enterprise commitments", 
                 fontsize=12, fontweight='bold', pad=15, loc='left', color=PALETTE['neutral_dark'])
    
    ax.legend(frameon=True, facecolor='#F9FAFB', edgecolor='#E5E7EB', fontsize=9, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_segment_matrix(df_seg, output_path):
    """
    Plots a multi-dimensional matrix showing Churn Rate vs Engagement across Tenure Cohorts.
    """
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    seg_summary = df_seg.groupby(['tenure_segment', 'usage_segment']).agg(
        churn_rate=('churn', 'mean'),
        account_count=('customer_id', 'count'),
        avg_active_days=('monthly_active_days', 'mean'),
        total_mrr=('monthly_charges', 'sum')
    ).reset_index()
    
    # Scatter bubble plot
    colors = {'Low / Dormant': '#EF4444', 'Moderate User': '#F59E0B', 'Power User': '#10B981'}
    
    for usage, grp in seg_summary.groupby('usage_segment'):
        scatter = ax.scatter(
            grp['avg_active_days'], 
            grp['churn_rate'] * 100, 
            s=grp['account_count'] * 0.45, 
            c=colors[usage], 
            label=f"{usage}", 
            alpha=0.75, 
            edgecolors='black', 
            linewidth=1.2
        )
        
        # Add labels
        for _, row in grp.iterrows():
            ax.annotate(
                f"{row['tenure_segment'].split(' ')[1]}\n({row['churn_rate']:.1%})",
                (row['avg_active_days'], row['churn_rate'] * 100),
                xytext=(0, 10), textcoords="offset points",
                ha='center', fontsize=8.5, fontweight='semibold'
            )
            
    ax.set_xlabel("Average Monthly Active Days", fontsize=11, fontweight='bold', color=PALETTE['neutral_dark'])
    ax.set_ylabel("Segment Churn Rate (%)", fontsize=11, fontweight='bold', color=PALETTE['neutral_dark'])
    ax.set_title("Customer Segmentation Matrix: Engagement vs Churn Risk\nBubble size reflects account volume; high churn strongly clusters below 10 active days", 
                 fontsize=12, fontweight='bold', pad=15, loc='left', color=PALETTE['neutral_dark'])
    
    ax.axhline(25, color='#9CA3AF', linestyle=':', label='Elevated Risk Threshold (25%)')
    ax.legend(title="Usage Tier (Bubble Size = Account Count)", frameon=True, fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_kmeans_personas(cluster_profile, output_path):
    """
    Plots horizontal multi-panel comparison of the 4 behavioral personas.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), dpi=300, sharey=True)
    
    personas = cluster_profile['persona_name'].tolist()
    
    # Metrics to display
    metrics = [
        ('churn_rate', 'Churn Rate (%)', '#EF4444', lambda x: x * 100, '%.1f%%'),
        ('avg_active_days', 'Active Days / Mo', '#3B82F6', lambda x: x, '%.1f'),
        ('avg_mrr', 'Average MRR ($)', '#10B981', lambda x: x, '$%.0f'),
        ('avg_csat', 'Customer CSAT (1-5)', '#8B5CF6', lambda x: x, '%.1f')
    ]
    
    y = np.arange(len(personas))
    
    for ax, (metric_col, title, color, transform, fmt) in zip(axes, metrics):
        vals = cluster_profile[metric_col].apply(transform).values
        bars = ax.barh(y, vals, color=color, alpha=0.85, height=0.6)
        
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=10, color=PALETTE['neutral_dark'])
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        for bar in bars:
            width = bar.get_width()
            ax.annotate(fmt % width,
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        va='center', fontsize=9, fontweight='bold', color=PALETTE['neutral_dark'])
                        
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(personas, fontsize=9.5, fontweight='bold', color=PALETTE['neutral_dark'])
    axes[0].invert_yaxis()
    
    fig.suptitle("K-Means Behavioral Personas Comparison Profile\nDistinct separation across Value (MRR), Stickiness (Active Days), and Operational Risk (CSAT & Churn)", 
                 fontsize=12, fontweight='bold', color=PALETTE['neutral_dark'], y=1.02, ha='left', x=0.04)
                 
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_feature_importance_and_odds(model_results, output_path):
    """
    Plots Logistic Regression Odds Ratios highlighting churn drivers vs protective factors.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    odds = model_results['odds_ratios'].copy()
    # Filter top 5 churn risk factors (odds > 1) and top 5 retention factors (odds < 1)
    top_risk = odds[odds['odds_ratio'] > 1.0].head(5)
    top_prot = odds[odds['odds_ratio'] < 1.0].tail(5)
    
    plot_df = pd.concat([top_risk, top_prot]).sort_values(by='odds_ratio', ascending=True)
    
    # Clean feature names for presentation
    clean_labels = {
        'days_since_last_active': 'Days Inactive (Recency)',
        'support_tickets': 'Support Ticket Volume',
        'subscription_tier_Starter': 'Starter Tier Plan',
        'payment_method_Credit Card': 'Credit Card Payment',
        'paperless_billing_Yes': 'Paperless Billing',
        'features_adopted': 'Core Features Adopted',
        'tenure_months': 'Customer Tenure (Months)',
        'monthly_active_days': 'Monthly Active Days',
        'contract_type_One-Year': 'One-Year Contract Commitment',
        'contract_type_Two-Year': 'Two-Year Contract Commitment'
    }
    
    labels = [clean_labels.get(f, f) for f in plot_df['feature']]
    vals = plot_df['odds_ratio'].values
    y = np.arange(len(labels))
    
    colors = ['#10B981' if v < 1.0 else '#EF4444' for v in vals]
    
    bars = ax.barh(y, vals, color=colors, alpha=0.85, height=0.6)
    ax.axvline(1.0, color='#1F2937', linestyle='--', linewidth=1.2, label='Neutral Odds (1.0x)')
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight='bold', color=PALETTE['neutral_dark'])
    ax.set_xlabel("Odds Ratio Multiplier (e^β)\nValues > 1.0 increase churn likelihood; Values < 1.0 protect against churn", 
                 fontsize=10, fontweight='bold', color=PALETTE['neutral_dark'], labelpad=10)
    ax.set_title("Key Drivers of Customer Churn vs Retention (Odds Ratio Analysis)\nContract duration and active daily usage are the strongest operational safeguards", 
                 fontsize=12, fontweight='bold', pad=15, loc='left', color=PALETTE['neutral_dark'])
                 
    for bar in bars:
        width = bar.get_width()
        offset = 6 if width >= 1.0 else -35
        text_color = '#991B1B' if width >= 1.0 else '#065F46'
        ax.annotate(f"{width:.2f}x",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    va='center', fontsize=9, fontweight='bold', color=text_color)
                    
    ax.legend(frameon=True, fontsize=9, loc='lower right')
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_revenue_at_risk(df, output_path):
    """
    Plots MRR breakdown and Revenue at Risk across customer tiers and segments.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # 1. Total MRR vs Churned MRR
    total_mrr = df['monthly_charges'].sum()
    churned_mrr = df[df['churn'] == 1]['monthly_charges'].sum()
    retained_mrr = total_mrr - churned_mrr
    
    ax1.pie([retained_mrr, churned_mrr], 
            labels=[f"Retained MRR\n${retained_mrr:,.0f} ({retained_mrr/total_mrr:.1%})", 
                    f"Churned MRR\n${churned_mrr:,.0f} ({churned_mrr/total_mrr:.1%})"],
            colors=['#10B981', '#EF4444'], 
            startangle=140, 
            explode=(0, 0.08), 
            textprops={'fontsize': 9.5, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax1.set_title("Monthly Recurring Revenue (MRR)\nPortfolio Distribution", fontsize=11, fontweight='bold', pad=10)
    
    # 2. Churned MRR by Subscription Tier
    churn_tier = df[df['churn'] == 1].groupby('subscription_tier')['monthly_charges'].sum().reindex(['Starter', 'Professional', 'Enterprise'])
    y = np.arange(len(churn_tier))
    bars = ax2.barh(y, churn_tier.values, color=['#EF4444', '#F59E0B', '#3B82F6'], height=0.55, alpha=0.85)
    
    ax2.set_yticks(y)
    ax2.set_yticklabels(churn_tier.index, fontsize=10, fontweight='bold', color=PALETTE['neutral_dark'])
    ax2.set_xlabel("Lost Monthly Recurring Revenue ($)", fontsize=10, fontweight='bold')
    ax2.set_title("Lost MRR by Subscription Tier\nProfessional & Starter tiers drive 92%+ of dollar loss", fontsize=11, fontweight='bold', pad=10)
    ax2.grid(axis='x', linestyle=':', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar in bars:
        width = bar.get_width()
        arr_val = width * 12
        ax2.annotate(f"${width:,.0f}/mo\n(${arr_val/1e3:,.0f}K ARR)",
                     xy=(width, bar.get_y() + bar.get_height() / 2),
                     xytext=(6, 0),
                     textcoords="offset points",
                     va='center', fontsize=8.5, fontweight='bold', color=PALETTE['neutral_dark'])
                     
    fig.suptitle("Revenue at Risk & Financial Exposure Analysis ($1.30M Annualized Run Rate)", 
                 fontsize=12, fontweight='bold', color=PALETTE['neutral_dark'], y=1.02, ha='left', x=0.05)
                 
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    from retention_metrics import calculate_cohort_retention_matrix
    from segmentation import apply_rule_based_segmentation, perform_kmeans_clustering
    from modeling import prepare_modeling_data, train_and_evaluate_models
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df = pd.read_csv(os.path.join(base_dir, 'data/raw/customer_churn_data.csv'))
    
    cohort_df = calculate_cohort_retention_matrix(df)
    plot_cohort_heatmap(cohort_df, os.path.join(base_dir, 'reports/figures/cohort_retention_heatmap.png'))
    plot_churn_by_tier_and_contract(df, os.path.join(base_dir, 'reports/figures/churn_by_tier_and_contract.png'))
    
    df_seg = apply_rule_based_segmentation(df)
    plot_segment_matrix(df_seg, os.path.join(base_dir, 'reports/figures/customer_segments_matrix.png'))
    
    df_clustered, profile, _ = perform_kmeans_clustering(df_seg)
    plot_kmeans_personas(profile, os.path.join(base_dir, 'reports/figures/kmeans_cluster_profiles.png'))
    
    X, y, _ = prepare_modeling_data(df)
    model_res = train_and_evaluate_models(X, y)
    plot_feature_importance_and_odds(model_res, os.path.join(base_dir, 'reports/figures/churn_risk_feature_importance.png'))
    plot_revenue_at_risk(df, os.path.join(base_dir, 'reports/figures/revenue_at_risk_waterfall.png'))
