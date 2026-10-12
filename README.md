# 📊 Customer Retention and Churn Analysis
### Multi-Dimensional Segmentation, Predictive Risk Scoring & Business Storytelling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![Domain](https://img.shields.io/badge/Domain-B2B%20SaaS%20%7C%20RevOps-purple.svg)]()

> **Portfolio Project 3**: A comprehensive customer retention diagnostics framework for a B2B SaaS platform (*SaaSify Cloud*). Analyzes behavioral engagement, lifecycle tenure, and contract structures across **7,500 customer accounts** ($9.11M ARR) to identify root-cause churn drivers, isolate high-risk segments, and present evidence-based, executive-ready retention playbooks to protect **$1.30M ARR**.

---

## 📑 Table of Contents
- [Executive Summary & Top-Line KPIs](#-executive-summary--top-line-kpis)
- [Key Business Insights](#-key-business-insights)
- [Cohort Retention & Lifecycle Dynamics](#-cohort-retention--lifecycle-dynamics)
- [Customer Segmentation Framework](#-customer-segmentation-framework)
  - [1. Multi-Dimensional Business Matrix](#1-multi-dimensional-business-matrix)
  - [2. Unsupervised Behavioral Personas (K-Means)](#2-unsupervised-behavioral-personas-k-means)
- [Predictive Risk Modeling & Odds Ratios](#-predictive-risk-modeling--odds-ratios)
- [Financial Exposure & Revenue at Risk](#-financial-exposure--revenue-at-risk)
- [Evidence-Based Strategic Recommendations (ROI Playbook)](#-evidence-based-strategic-recommendations-roi-playbook)
- [Repository Structure](#-repository-structure)
- [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
- [Skills & Competencies Demonstrated](#-skills--competencies-demonstrated)

---

## 📈 Executive Summary & Top-Line KPIs

In recurring revenue businesses, retaining existing accounts is the primary engine of sustainable compounding. Acquiring a new customer costs **5x to 7x more** than retaining an existing one. Early customer churn destroys Customer Acquisition Cost (CAC) payback, severely depressing customer Lifetime Value (LTV).

| Metric | Portfolio Value | Benchmark | Strategic Status |
| :--- | :--- | :--- | :--- |
| **Total Analyzed Accounts** | **7,500** | - | Complete 2-Year Dataset |
| **Overall Logo Retention Rate** | **73.76%** | 80% – 85% | ⚠️ Below Target |
| **Overall Logo Churn Rate** | **26.24%** (1,968 accounts) | 15% – 20% | ⚠️ Elevated Churn |
| **Total Portfolio MRR** | **$759,409 / mo** (~$9.11M ARR) | - | Portfolio Base |
| **Gross MRR Churn Rate** | **14.29%** ($108,504 / mo) | 8% – 10% | ⚠️ High Dollar Leakage |
| **Annualized Revenue at Risk** | **$1,302,045 / yr** | - | 🚨 Priority Intervention Area |
| **Average Revenue Per User (ARPU)** | **$101.25 / mo** | - | Healthy Account Economics |
| **Mean Customer Tenure** | **12.5 months** | 24+ months | Sub-optimal Lifespan |

```
💡 Core Takeaway:
Logo Churn (26.2%) is nearly DOUBLE MRR Churn (14.3%).
Enterprise customers ($299/mo) are exceptionally sticky (3.4% churn).
The core revenue leak is concentrated in high-velocity, month-to-month Starter and Pro accounts that fail to cross product activation thresholds within their first 90 days.
```

---

## 🔍 Key Business Insights

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE 4 CORE CHURN DRIVERS                                │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 1. The Onboarding Cliff  │ 2. Support Friction      │ 3. Feature Threshold Lock-in     │
│ Months 1-3 Starter churn │ ≥3 tickets & CSAT < 3    │ <3 features: 59.8% churn         │
│ reaches 41.0%; drop-off  │ increase churn odds by   │ vs 11.2% for accounts adopting   │
│ concentrates in first 90d│ 2.4x.                    │ 4+ core product workflows.       │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

1. **The "Onboarding Cliff" (Months 1–3)**:  
   New customers experience their highest attrition within the first 90 days of signup. Cohort survival drops to 74% by Month 3 before stabilizing at 88%+ for accounts surviving past Month 6.
2. **Contract Type as a Retention Shield**:  
   Month-to-Month contracts churn at **36.5%**, compared to **15.6%** for One-Year contracts and **10.0%** for Two-Year contracts. Annual commitments cut churn odds by **60%** ($\text{Odds Ratio} = 0.40$).
3. **The Feature Adoption Tipping Point**:  
   Accounts adopting fewer than 3 features suffer a **59.8% churn rate**. Accounts with 4+ features drop to **11.2%**, and 6+ features fall to **2.5%**. Each additional feature decreases churn odds by **38.4%** ($\text{Odds Ratio} = 0.616$).
4. **Customer Support as an Early Warning Indicator**:  
   Low ticket volume (0–1 tickets) indicates healthy self-service (15.8% churn). However, accounts opening **4+ tickets experience 51.2% churn**, signaling unaddressed technical roadblocks and product frustration.

---

## 🗓️ Cohort Retention & Lifecycle Dynamics

Cohort analysis groups accounts by their signup month and tracks active retention across elapsed tenure months.

<p align="center">
  <img src="reports/figures/cohort_retention_heatmap.png" width="900" alt="Cohort Retention Matrix" />
</p>

- **Onboarding Vulnerability**: Across all 2024–2025 cohorts, the sharpest attrition occurs in **M1–M3** (average retention drops ~26 percentage points).
- **Long-Term Stickiness**: Customers who navigate past Month 6 exhibit steady retention above 85–90%, proving that product value is high once customer onboarding is successfully completed.

---

## 🧩 Customer Segmentation Framework

### 1. Multi-Dimensional Business Matrix
We segmented the portfolio across three strategic axes:
- **Tenure Lifecycle**: Onboarding (1–3m), Growth (4–12m), Mature (13–24m)
- **Usage Intensity**: Low/Dormant (≤8 active days OR ≤2 features), Moderate, Power (≥19 active days AND ≥5 features)
- **Subscription Tier**: Starter ($29), Professional ($99), Enterprise ($299)

<p align="center">
  <img src="reports/figures/churn_by_tier_and_contract.png" width="850" alt="Churn by Tier and Contract" />
</p>

<p align="center">
  <img src="reports/figures/customer_segments_matrix.png" width="900" alt="Customer Segmentation Matrix" />
</p>

#### Top High-Churn Segments (Critical Action Required):
| Segment (Tier \| Tenure \| Usage) | Accounts | Churn Rate | Lost MRR | ARR at Risk | Primary Driver |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Starter \| Onboarding (1-3m) \| Low/Dormant** | 682 | **63.49%** | $12,240 | $146,880 | Poor activation, zero guided setup |
| **Starter \| Growth (4-12m) \| Low/Dormant** | 594 | **58.75%** | $10,120 | $121,440 | Fading product utility, month-to-month |
| **Professional \| Onboarding (1-3m) \| Low/Dormant** | 310 | **54.84%** | $17,424 | $209,088 | High price relative to perceived value |
| **Professional \| Growth (4-12m) \| Low/Dormant** | 412 | **48.06%** | $20,394 | $244,728 | Unresolved technical blockers, low login |

---

### 2. Unsupervised Behavioral Personas (K-Means)
Using normalized behavioral and engagement features (`tenure_months`, `monthly_charges`, `monthly_active_days`, `features_adopted`, `api_calls_monthly`, `support_tickets`, `csat_score`, `days_since_last_active`), K-Means clustering revealed 4 distinct customer personas (Silhouette Score: **0.225**):

<p align="center">
  <img src="reports/figures/kmeans_cluster_profiles.png" width="950" alt="K-Means Behavioral Personas" />
</p>

| Persona | Account Share | Churn Rate | Avg MRR | Active Days | CSAT | ARR at Risk | Strategic Posture |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 🏆 **Enterprise Powerhouses** | 14.1% (1,054) | **2.47%** | $312 | 21.7 | 4.29 | $97,233 | **Nurture & Expand**: Dedicated CSM, Executive Business Reviews |
| 🛡️ **Satisfied Core Adopters** | 42.1% (3,159) | **12.00%** | $74 | 16.9 | 4.49 | $267,479 | **Upsell & Lock-in**: Convert monthly contracts to annual |
| ⚠️ **Support Friction / Stalled** | 19.9% (1,490) | **32.82%** | $73 | 14.1 | 3.06 | $362,260 | **Triage & Resolve**: Urgent ticket escalation & CSM check-ins |
| 🚨 **Dormant Flight Risks** | 24.0% (1,797) | **59.77%** | $48 | 5.6 | 3.81 | $575,073 | **Re-engage**: Automated email drips, self-serve onboarding |

---

## 🤖 Predictive Risk Modeling & Odds Ratios

We trained both **Logistic Regression** (for interpretable odds ratios) and **Random Forest** (for non-linear threshold modeling) to predict churn:
- **Logistic Regression ROC-AUC**: `0.8887`
- **Random Forest ROC-AUC**: `0.8726`

<p align="center">
  <img src="reports/figures/churn_risk_feature_importance.png" width="850" alt="Odds Ratio Feature Importance" />
</p>

### Econometric Odds Ratios ($e^\beta$):
- **Days Inactive (Recency)**: $\text{Odds Ratio} = 1.57\times$ (+57% churn odds per standard deviation increase in dormancy).
- **Support Tickets**: $\text{Odds Ratio} = 1.28\times$ (+28% churn odds per increase in ticket volume).
- **Two-Year Contract**: $\text{Odds Ratio} = 0.39\times$ (**61% reduction in churn odds**).
- **One-Year Contract**: $\text{Odds Ratio} = 0.40\times$ (**60% reduction in churn odds**).
- **Monthly Active Days**: $\text{Odds Ratio} = 0.41\times$ (**59% reduction in churn odds**).
- **Core Features Adopted**: $\text{Odds Ratio} = 0.62\times$ (**38% reduction in churn odds per feature**).

---

## 💰 Financial Exposure & Revenue at Risk

<p align="center">
  <img src="reports/figures/revenue_at_risk_waterfall.png" width="900" alt="Revenue at Risk Waterfall" />
</p>

- **Total Lost MRR**: **$108,504 / month** ($\$1,302,045$ Annualized Run Rate).
- **Loss Concentration**: Professional Tier accounts represent **$51,800/mo ($621K ARR)** and Starter Tier accounts represent **$48,600/mo ($583K ARR)**. Enterprise churn accounts for only $8,100/mo ($97K ARR).

---

## 🎯 Evidence-Based Strategic Recommendations (ROI Playbook)

Based on empirical modeling, we recommend four prioritized initiatives with projected financial impact:

```
                            PROJECTED ANNUAL FINANCIAL IMPACT
┌─────────────────────────────────┬──────────────┬───────────────┬───────────────┐
│ Strategic Initiative            │ Target Group │ Projected     │ Annualized    │
│                                 │              │ Churn Drop    │ ARR Protected │
├─────────────────────────────────┼──────────────┼───────────────┼───────────────┤
│ 1. "First 30 Days" Onboarding   │ Months 1-3   │ -18% in Early │ +$195,000     │
│    Activation Architecture      │ Starter/Pro  │ Churn         │               │
│ 2. Proactive Support CS Triage  │ CSAT < 3     │ -25% in       │ +$130,000     │
│    & Ticket Velocity Alerting   │ Tickets ≥ 3  │ Friction Drop │               │
│ 3. Annual Contract Migration    │ Month-to-    │ -40% in Cohort│ +$145,000     │
│    Campaign (2 Months Free)     │ Month Users  │ Churn         │               │
│ 4. Automated Dormancy "Pulse"   │ Inactive     │ -12% in       │ +$75,000      │
│    In-App Re-engagement         │ > 14 Days    │ Dormant Drop  │               │
├─────────────────────────────────┴──────────────┴───────────────┴───────────────┤
│ TOTAL PROJECTED ANNUAL IMPACT:                                  +$545,000 ARR  │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 1. 🚀 "First 30 Days" Onboarding Activation Architecture
- **Problem**: 63% churn among new Starter accounts with low usage.
- **Action**: Deploy an interactive, milestone-based in-app onboarding checklist. Users who connect 3 core workflows within 14 days unlock complimentary feature credits.
- **Target**: Increase 14-day 3-feature adoption rate from **28% to 65%**.
- **Financial Return**: Protects **+$195,000 ARR**.

### 2. 🛡️ Proactive Support CS Triage & Red-Account Alerting
- **Problem**: Accounts logging ≥3 tickets churn at 48%+.
- **Action**: Establish an automated real-time webhook connecting Zendesk/Intercom to Slack and Salesforce. Any account logging their 3rd ticket within 30 days is automatically flagged as a **"Red Account"** for senior CSM outreach within 24 hours.
- **Target**: Reduce repeat ticket cycle time by 40%; lift CSAT from 3.06 to 4.2+.
- **Financial Return**: Protects **+$130,000 ARR**.

### 3. 📄 Annual Contract Incentive Campaign ("Save 17%")
- **Problem**: Month-to-month accounts churn at 36.5% vs 15.6% for annual commitments.
- **Action**: Launch an in-app promotion offering **2 months free** (16.7% discount) for converting to an annual upfront contract, targeted specifically at healthy month-to-month accounts (>15 active days/mo).
- **Target**: Convert 25% of active monthly accounts to annual agreements within 6 months.
- **Financial Return**: Locks in **+$145,000 ARR**.

### 4. ⚡ Automated Dormancy "Pulse Check"
- **Problem**: Churn probability exceeds 60% after 14 days of inactivity.
- **Action**: Trigger automated behavioral nurture emails at Day 10 of inactivity ("Here's what your team achieved this month"), followed by a personal CSM check-in at Day 18.
- **Target**: Re-activate 15% of dormant accounts before billing cycle renewal.
- **Financial Return**: Protects **+$75,000 ARR**.

---

## 📂 Repository Structure

```
customer-retention-churn-analysis/
├── README.md                               # Comprehensive Project Overview & Executive Brief
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusion patterns
├── LICENSE                                 # MIT License
├── data/
│   ├── raw/
│   │   └── customer_churn_data.csv         # 7,500 customer records (reproducible seed=42)
│   └── processed/
│       └── customer_segments_scored.csv    # Enriched dataset with segments, clusters & risk scores
├── notebooks/
│   └── customer_retention_churn_analysis.ipynb # Complete interactive Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── data_generator.py                   # Realistic B2B SaaS synthetic data generator
│   ├── retention_metrics.py                # Logo/MRR churn, cohort triangles & LTV engine
│   ├── segmentation.py                     # Rule-based matrix & K-Means clustering
│   ├── modeling.py                         # Predictive risk scoring & odds-ratio analysis
│   ├── visualizations.py                   # Publication-grade chart generation suite
│   └── run_pipeline.py                     # Master execution pipeline script
└── reports/
    ├── executive_summary.md                # C-level executive briefing memo
    └── figures/                            # High-resolution saved charts (300 DPI)
        ├── cohort_retention_heatmap.png
        ├── churn_by_tier_and_contract.png
        ├── customer_segments_matrix.png
        ├── kmeans_cluster_profiles.png
        ├── churn_risk_feature_importance.png
        └── revenue_at_risk_waterfall.png
```

---

## 🚀 Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/customer-retention-churn-analysis.git
cd customer-retention-churn-analysis
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Master End-to-End Pipeline
Executes data generation, metrics computation, customer segmentation, ML risk scoring, and chart rendering:
```bash
python3 src/run_pipeline.py
```

### 4. Launch the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/customer_retention_churn_analysis.ipynb
```

---

## 🧠 Skills & Competencies Demonstrated

- **Analytical Thinking**: Translating raw behavioral logs into unit economics, cohort triangles, and survival curves.
- **Multi-Dimensional Segmentation**: Combining rule-based business logic (Tenure × Usage × Plan) with unsupervised machine learning (K-Means).
- **Statistical & ML Modeling**: Evaluating predictive classification models (Logistic Regression, Random Forest), calculating Odds Ratios ($e^\beta$), and quantifying ROC-AUC performance.
- **Business Storytelling & Executive Communication**: Framing data science insights around Revenue at Risk, CAC Payback, ARPU, and actionable, prioritized C-suite ROI playbooks.
- **Production-Grade Engineering**: Modular code organization (`src/`), reproducible random states, comprehensive documentation, and presentation-grade visualizations.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
