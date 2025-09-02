-- Advanced Cohort Analysis and Retention Metrics
-- Demonstrates sophisticated time-series analysis and customer lifecycle tracking

-- =====================================================
-- 1. MONTHLY COHORT ANALYSIS
-- =====================================================

-- Create monthly signup cohorts and track retention
WITH monthly_cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', signup_date) as cohort_month,
        signup_date,
        cleaned_monthly_revenue,
        standardized_subscription_tier,
        activation_completed,
        cleaned_churn_risk_score,
        total_sessions,
        days_since_signup
    FROM cleaned_customer_data
    WHERE signup_date >= '2023-01-01'  -- Focus on recent cohorts
),

-- Calculate cohort sizes and initial metrics
cohort_summary AS (
    SELECT 
        cohort_month,
        COUNT(*) as cohort_size,
        COUNT(CASE WHEN activation_completed = 1 THEN 1 END) as activated_count,
        SUM(cleaned_monthly_revenue) as cohort_revenue,
        AVG(cleaned_monthly_revenue) as avg_revenue_per_customer,
        ROUND(100.0 * COUNT(CASE WHEN activation_completed = 1 THEN 1 END) / COUNT(*), 2) as activation_rate,
        AVG(cleaned_churn_risk_score) as avg_initial_churn_risk
    FROM monthly_cohorts
    GROUP BY cohort_month
),

-- Simulate retention tracking (in real scenario, this would use actual activity data)
retention_simulation AS (
    SELECT 
        mc.cohort_month,
        mc.customer_id,
        mc.cleaned_monthly_revenue,
        mc.cleaned_churn_risk_score,
        
        -- Simulate retention based on churn risk and engagement
        CASE 
            WHEN mc.cleaned_churn_risk_score <= 0.2 THEN 1  -- Very likely to retain
            WHEN mc.cleaned_churn_risk_score <= 0.4 THEN 
                CASE WHEN RANDOM() > 0.1 THEN 1 ELSE 0 END  -- 90% retention
            WHEN mc.cleaned_churn_risk_score <= 0.6 THEN 
                CASE WHEN RANDOM() > 0.3 THEN 1 ELSE 0 END  -- 70% retention
            WHEN mc.cleaned_churn_risk_score <= 0.8 THEN 
                CASE WHEN RANDOM() > 0.6 THEN 1 ELSE 0 END  -- 40% retention
            ELSE 
                CASE WHEN RANDOM() > 0.8 THEN 1 ELSE 0 END  -- 20% retention
        END as retained_month_1,
        
        -- Additional retention periods
        CASE 
            WHEN mc.cleaned_churn_risk_score <= 0.3 THEN 
                CASE WHEN RANDOM() > 0.05 THEN 1 ELSE 0 END  -- 95% of month 1 survivors
            WHEN mc.cleaned_churn_risk_score <= 0.6 THEN 
                CASE WHEN RANDOM() > 0.2 THEN 1 ELSE 0 END   -- 80% of month 1 survivors
            ELSE 
                CASE WHEN RANDOM() > 0.4 THEN 1 ELSE 0 END   -- 60% of month 1 survivors
        END as retained_month_3,
        
        CASE 
            WHEN mc.cleaned_churn_risk_score <= 0.4 THEN 
                CASE WHEN RANDOM() > 0.1 THEN 1 ELSE 0 END   -- 90% of month 3 survivors
            ELSE 
                CASE WHEN RANDOM() > 0.3 THEN 1 ELSE 0 END   -- 70% of month 3 survivors
        END as retained_month_6
        
    FROM monthly_cohorts mc
)

-- Generate cohort retention table
SELECT 
    cs.cohort_month,
    cs.cohort_size,
    cs.activated_count,
    cs.activation_rate,
    ROUND(cs.cohort_revenue, 2) as initial_cohort_revenue,
    ROUND(cs.avg_revenue_per_customer, 2) as avg_revenue_per_customer,
    
    -- Retention metrics
    SUM(rs.retained_month_1) as retained_month_1,
    SUM(rs.retained_month_3) as retained_month_3,
    SUM(rs.retained_month_6) as retained_month_6,
    
    -- Retention rates
    ROUND(100.0 * SUM(rs.retained_month_1) / cs.cohort_size, 2) as retention_rate_month_1,
    ROUND(100.0 * SUM(rs.retained_month_3) / cs.cohort_size, 2) as retention_rate_month_3,
    ROUND(100.0 * SUM(rs.retained_month_6) / cs.cohort_size, 2) as retention_rate_month_6,
    
    -- Revenue retention
    ROUND(SUM(CASE WHEN rs.retained_month_1 = 1 THEN rs.cleaned_monthly_revenue ELSE 0 END), 2) as revenue_month_1,
    ROUND(SUM(CASE WHEN rs.retained_month_3 = 1 THEN rs.cleaned_monthly_revenue ELSE 0 END), 2) as revenue_month_3,
    ROUND(SUM(CASE WHEN rs.retained_month_6 = 1 THEN rs.cleaned_monthly_revenue ELSE 0 END), 2) as revenue_month_6,
    
    -- Revenue retention rates
    ROUND(100.0 * SUM(CASE WHEN rs.retained_month_1 = 1 THEN rs.cleaned_monthly_revenue ELSE 0 END) / cs.cohort_revenue, 2) as revenue_retention_month_1,
    ROUND(100.0 * SUM(CASE WHEN rs.retained_month_3 = 1 THEN rs.cleaned_monthly_revenue ELSE 0 END) / cs.cohort_revenue, 2) as revenue_retention_month_3,
    ROUND(100.0 * SUM(CASE WHEN rs.retained_month_6 = 1 THEN rs.cleaned_monthly_revenue ELSE 0 END) / cs.cohort_revenue, 2) as revenue_retention_month_6

FROM cohort_summary cs
JOIN retention_simulation rs ON cs.cohort_month = rs.cohort_month
GROUP BY cs.cohort_month, cs.cohort_size, cs.activated_count, cs.activation_rate, cs.cohort_revenue, cs.avg_revenue_per_customer
ORDER BY cs.cohort_month;

-- =====================================================
-- 2. SUBSCRIPTION TIER COHORT ANALYSIS
-- =====================================================

-- Analyze retention patterns by subscription tier
WITH tier_cohorts AS (
    SELECT 
        standardized_subscription_tier,
        DATE_TRUNC('month', signup_date) as cohort_month,
        COUNT(*) as cohort_size,
        AVG(cleaned_monthly_revenue) as avg_initial_revenue,
        AVG(cleaned_churn_risk_score) as avg_churn_risk,
        COUNT(CASE WHEN activation_completed = 1 THEN 1 END) as activated_customers
    FROM cleaned_customer_data
    WHERE signup_date >= '2023-01-01'
    GROUP BY standardized_subscription_tier, DATE_TRUNC('month', signup_date)
),

-- Calculate tier-specific retention expectations
tier_retention_analysis AS (
    SELECT 
        tc.*,
        -- Expected retention based on tier (Premium/Enterprise typically have better retention)
        CASE 
            WHEN tc.standardized_subscription_tier = 'Enterprise' THEN 0.95
            WHEN tc.standardized_subscription_tier = 'Premium' THEN 0.85
            ELSE 0.70
        END as expected_month_1_retention,
        
        CASE 
            WHEN tc.standardized_subscription_tier = 'Enterprise' THEN 0.90
            WHEN tc.standardized_subscription_tier = 'Premium' THEN 0.75
            ELSE 0.55
        END as expected_month_6_retention,
        
        -- Adjust for churn risk
        CASE 
            WHEN tc.standardized_subscription_tier = 'Enterprise' THEN 0.95 * (1 - tc.avg_churn_risk * 0.3)
            WHEN tc.standardized_subscription_tier = 'Premium' THEN 0.85 * (1 - tc.avg_churn_risk * 0.4)
            ELSE 0.70 * (1 - tc.avg_churn_risk * 0.5)
        END as risk_adjusted_retention
        
    FROM tier_cohorts tc
)

SELECT 
    standardized_subscription_tier,
    cohort_month,
    cohort_size,
    activated_customers,
    ROUND(100.0 * activated_customers / cohort_size, 2) as activation_rate,
    ROUND(avg_initial_revenue, 2) as avg_initial_revenue,
    ROUND(avg_churn_risk, 3) as avg_churn_risk,
    ROUND(expected_month_1_retention * 100, 1) as expected_month_1_retention_pct,
    ROUND(expected_month_6_retention * 100, 1) as expected_month_6_retention_pct,
    ROUND(risk_adjusted_retention * 100, 1) as risk_adjusted_retention_pct,
    
    -- Estimated customer lifetime value by tier
    ROUND(avg_initial_revenue * 12 * risk_adjusted_retention * 2, 2) as estimated_24_month_ltv

FROM tier_retention_analysis
ORDER BY standardized_subscription_tier, cohort_month;

-- =====================================================
-- 3. ADVANCED RETENTION METRICS AND CHURN PREDICTION
-- =====================================================

-- Create comprehensive retention and churn analysis
WITH customer_lifecycle_metrics AS (
    SELECT 
        customer_id,
        signup_date,
        standardized_subscription_tier,
        standardized_industry,
        days_since_signup,
        cleaned_monthly_revenue,
        total_sessions,
        cleaned_churn_risk_score,
        cleaned_satisfaction_score,
        activation_completed,
        
        -- Lifecycle stage based on days since signup
        CASE 
            WHEN days_since_signup <= 7 THEN 'Week 1'
            WHEN days_since_signup <= 30 THEN 'Month 1'
            WHEN days_since_signup <= 90 THEN 'Month 2-3'
            WHEN days_since_signup <= 180 THEN 'Month 4-6'
            WHEN days_since_signup <= 365 THEN 'Month 7-12'
            ELSE 'Year 2+'
        END as lifecycle_stage,
        
        -- Engagement level
        CASE 
            WHEN total_sessions = 0 THEN 'No Usage'
            WHEN total_sessions <= 5 THEN 'Low Usage'
            WHEN total_sessions <= 20 THEN 'Medium Usage'
            WHEN total_sessions <= 50 THEN 'High Usage'
            ELSE 'Power User'
        END as usage_level,
        
        -- Revenue tier
        NTILE(4) OVER (ORDER BY cleaned_monthly_revenue) as revenue_quartile,
        
        -- Churn risk category
        CASE 
            WHEN cleaned_churn_risk_score <= 0.3 THEN 'Low Risk'
            WHEN cleaned_churn_risk_score <= 0.6 THEN 'Medium Risk'
            WHEN cleaned_churn_risk_score <= 0.8 THEN 'High Risk'
            ELSE 'Critical Risk'
        END as churn_risk_category
        
    FROM cleaned_customer_data
),

-- Aggregate retention insights
retention_insights AS (
    SELECT 
        lifecycle_stage,
        churn_risk_category,
        standardized_subscription_tier,
        
        COUNT(*) as customer_count,
        AVG(cleaned_monthly_revenue) as avg_revenue,
        AVG(total_sessions) as avg_sessions,
        AVG(cleaned_satisfaction_score) as avg_satisfaction,
        COUNT(CASE WHEN activation_completed = 1 THEN 1 END) as activated_count,
        
        -- Risk distribution
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY lifecycle_stage), 2) as pct_of_lifecycle_stage,
        
        -- Estimated retention probability (inverse of churn risk)
        AVG(1 - cleaned_churn_risk_score) as avg_retention_probability
        
    FROM customer_lifecycle_metrics
    GROUP BY lifecycle_stage, churn_risk_category, standardized_subscription_tier
)

SELECT 
    lifecycle_stage,
    churn_risk_category,
    standardized_subscription_tier,
    customer_count,
    ROUND(avg_revenue, 2) as avg_monthly_revenue,
    ROUND(avg_sessions, 1) as avg_sessions,
    ROUND(avg_satisfaction, 2) as avg_satisfaction,
    activated_count,
    ROUND(100.0 * activated_count / customer_count, 2) as activation_rate,
    pct_of_lifecycle_stage,
    ROUND(avg_retention_probability * 100, 1) as estimated_retention_rate,
    
    -- Business impact metrics
    ROUND(customer_count * avg_revenue, 2) as segment_monthly_revenue,
    ROUND(customer_count * avg_revenue * avg_retention_probability * 12, 2) as estimated_annual_retained_revenue

FROM retention_insights
WHERE customer_count >= 5  -- Filter out very small segments
ORDER BY lifecycle_stage, churn_risk_category, standardized_subscription_tier;

-- =====================================================
-- 4. COHORT PERFORMANCE SUMMARY
-- =====================================================

-- Executive summary of cohort performance
WITH cohort_performance_summary AS (
    SELECT 
        DATE_TRUNC('quarter', signup_date) as signup_quarter,
        standardized_subscription_tier,
        
        COUNT(*) as total_customers,
        SUM(cleaned_monthly_revenue) as total_revenue,
        AVG(cleaned_monthly_revenue) as avg_revenue_per_customer,
        
        -- Activation metrics
        COUNT(CASE WHEN activation_completed = 1 THEN 1 END) as activated_customers,
        ROUND(100.0 * COUNT(CASE WHEN activation_completed = 1 THEN 1 END) / COUNT(*), 2) as activation_rate,
        
        -- Risk assessment
        AVG(cleaned_churn_risk_score) as avg_churn_risk,
        COUNT(CASE WHEN cleaned_churn_risk_score > 0.7 THEN 1 END) as high_risk_customers,
        ROUND(100.0 * COUNT(CASE WHEN cleaned_churn_risk_score > 0.7 THEN 1 END) / COUNT(*), 2) as high_risk_rate,
        
        -- Satisfaction
        AVG(cleaned_satisfaction_score) as avg_satisfaction,
        
        -- Estimated LTV
        AVG(cleaned_monthly_revenue * 12 * (1 - cleaned_churn_risk_score) * 2) as estimated_24_month_ltv
        
    FROM cleaned_customer_data
    WHERE signup_date >= '2023-01-01'
    GROUP BY DATE_TRUNC('quarter', signup_date), standardized_subscription_tier
)

SELECT 
    signup_quarter,
    standardized_subscription_tier,
    total_customers,
    ROUND(total_revenue, 2) as total_monthly_revenue,
    ROUND(avg_revenue_per_customer, 2) as avg_revenue_per_customer,
    activated_customers,
    activation_rate,
    ROUND(avg_churn_risk, 3) as avg_churn_risk,
    high_risk_customers,
    high_risk_rate,
    ROUND(avg_satisfaction, 2) as avg_satisfaction,
    ROUND(estimated_24_month_ltv, 2) as estimated_24_month_ltv,
    
    -- Quarter-over-quarter growth
    ROUND(100.0 * (total_customers - LAG(total_customers) OVER (
        PARTITION BY standardized_subscription_tier 
        ORDER BY signup_quarter
    )) / NULLIF(LAG(total_customers) OVER (
        PARTITION BY standardized_subscription_tier 
        ORDER BY signup_quarter
    ), 0), 2) as customer_growth_qoq,
    
    ROUND(100.0 * (total_revenue - LAG(total_revenue) OVER (
        PARTITION BY standardized_subscription_tier 
        ORDER BY signup_quarter
    )) / NULLIF(LAG(total_revenue) OVER (
        PARTITION BY standardized_subscription_tier 
        ORDER BY signup_quarter
    ), 0), 2) as revenue_growth_qoq

FROM cohort_performance_summary
ORDER BY signup_quarter, standardized_subscription_tier;
