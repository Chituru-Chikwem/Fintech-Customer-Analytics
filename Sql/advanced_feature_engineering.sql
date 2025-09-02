-- Advanced Feature Engineering using SQL
-- Demonstrates complex analytical functions and business metric calculations

-- =====================================================
-- 1. CUSTOMER VALUE AND ENGAGEMENT METRICS
-- =====================================================

-- Create comprehensive customer value scoring
WITH customer_metrics AS (
    SELECT 
        customer_id,
        signup_date,
        cleaned_monthly_revenue,
        standardized_subscription_tier,
        total_sessions,
        avg_session_duration,
        api_calls_per_month,
        cleaned_feature_usage_score,
        cleaned_satisfaction_score,
        cleaned_churn_risk_score,
        days_since_signup,
        
        -- Revenue-based metrics
        cleaned_monthly_revenue * 12 as estimated_annual_revenue,
        CASE 
            WHEN cleaned_monthly_revenue = 0 THEN 0
            ELSE cleaned_monthly_revenue / NULLIF(total_sessions, 0)
        END as revenue_per_session,
        
        -- Engagement metrics
        CASE 
            WHEN total_sessions = 0 THEN 0
            ELSE api_calls_per_month / NULLIF(total_sessions, 0)
        END as api_calls_per_session,
        
        -- Usage intensity
        CASE 
            WHEN days_since_signup = 0 THEN total_sessions
            ELSE total_sessions / NULLIF(days_since_signup, 0)
        END as sessions_per_day,
        
        -- Customer health score (composite metric)
        (
            cleaned_feature_usage_score * 0.25 +
            (cleaned_satisfaction_score / 5.0) * 0.25 +
            (1 - cleaned_churn_risk_score) * 0.25 +
            LEAST(1.0, total_sessions / 100.0) * 0.25
        ) as customer_health_score
        
    FROM cleaned_customer_data
),

-- Calculate percentiles for relative scoring
percentile_benchmarks AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY estimated_annual_revenue) as revenue_q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY estimated_annual_revenue) as revenue_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY estimated_annual_revenue) as revenue_q3,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY estimated_annual_revenue) as revenue_p90,
        
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY customer_health_score) as health_q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY customer_health_score) as health_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY customer_health_score) as health_q3,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY customer_health_score) as health_p90
    FROM customer_metrics
)

-- Create final feature-engineered dataset
SELECT 
    cm.*,
    pb.revenue_median,
    pb.health_median,
    
    -- Revenue tier classification
    CASE 
        WHEN cm.estimated_annual_revenue >= pb.revenue_p90 THEN 'Top 10%'
        WHEN cm.estimated_annual_revenue >= pb.revenue_q3 THEN 'High Value'
        WHEN cm.estimated_annual_revenue >= pb.revenue_median THEN 'Medium Value'
        WHEN cm.estimated_annual_revenue >= pb.revenue_q1 THEN 'Low Value'
        ELSE 'Minimal Value'
    END as revenue_tier,
    
    -- Health tier classification
    CASE 
        WHEN cm.customer_health_score >= pb.health_p90 THEN 'Excellent'
        WHEN cm.customer_health_score >= pb.health_q3 THEN 'Good'
        WHEN cm.customer_health_score >= pb.health_median THEN 'Average'
        WHEN cm.customer_health_score >= pb.health_q1 THEN 'Poor'
        ELSE 'Critical'
    END as health_tier,
    
    -- Engagement level
    CASE 
        WHEN cm.sessions_per_day >= 2 THEN 'Highly Engaged'
        WHEN cm.sessions_per_day >= 0.5 THEN 'Moderately Engaged'
        WHEN cm.sessions_per_day >= 0.1 THEN 'Lightly Engaged'
        ELSE 'Inactive'
    END as engagement_level,
    
    -- Customer segment (combining revenue and health)
    CASE 
        WHEN cm.estimated_annual_revenue >= pb.revenue_q3 AND cm.customer_health_score >= pb.health_q3 THEN 'Champions'
        WHEN cm.estimated_annual_revenue >= pb.revenue_q3 AND cm.customer_health_score < pb.health_q3 THEN 'High-Value At-Risk'
        WHEN cm.estimated_annual_revenue < pb.revenue_q3 AND cm.customer_health_score >= pb.health_q3 THEN 'Potential Champions'
        WHEN cm.estimated_annual_revenue >= pb.revenue_median AND cm.customer_health_score >= pb.health_median THEN 'Solid Performers'
        WHEN cm.cleaned_churn_risk_score > 0.7 THEN 'At-Risk'
        ELSE 'Developing'
    END as customer_segment,
    
    -- Time-based features
    EXTRACT(MONTH FROM cm.signup_date) as signup_month,
    EXTRACT(QUARTER FROM cm.signup_date) as signup_quarter,
    EXTRACT(YEAR FROM cm.signup_date) as signup_year,
    
    -- Seasonal signup indicator
    CASE 
        WHEN EXTRACT(MONTH FROM cm.signup_date) IN (12, 1, 2) THEN 'Winter'
        WHEN EXTRACT(MONTH FROM cm.signup_date) IN (3, 4, 5) THEN 'Spring'
        WHEN EXTRACT(MONTH FROM cm.signup_date) IN (6, 7, 8) THEN 'Summer'
        ELSE 'Fall'
    END as signup_season

FROM customer_metrics cm
CROSS JOIN percentile_benchmarks pb;

-- =====================================================
-- 2. ADVANCED WINDOW FUNCTIONS FOR TREND ANALYSIS
-- =====================================================

-- Create customer journey and trend analysis
WITH customer_journey AS (
    SELECT 
        customer_id,
        signup_date,
        cleaned_monthly_revenue,
        total_sessions,
        customer_health_score,
        
        -- Ranking within subscription tier
        ROW_NUMBER() OVER (
            PARTITION BY standardized_subscription_tier 
            ORDER BY cleaned_monthly_revenue DESC
        ) as revenue_rank_in_tier,
        
        -- Percentile ranking overall
        PERCENT_RANK() OVER (ORDER BY cleaned_monthly_revenue) as revenue_percentile,
        PERCENT_RANK() OVER (ORDER BY customer_health_score) as health_percentile,
        
        -- Moving averages (simulated with LAG for demonstration)
        AVG(cleaned_monthly_revenue) OVER (
            ORDER BY signup_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as revenue_30day_avg,
        
        -- Cumulative metrics
        SUM(cleaned_monthly_revenue) OVER (
            ORDER BY signup_date 
            ROWS UNBOUNDED PRECEDING
        ) as cumulative_revenue,
        
        -- Growth indicators
        LAG(cleaned_monthly_revenue, 1) OVER (
            PARTITION BY customer_id 
            ORDER BY signup_date
        ) as previous_revenue,
        
        -- Cohort analysis preparation
        DATE_TRUNC('month', signup_date) as signup_month_cohort,
        
        -- Customer lifetime value estimation
        CASE 
            WHEN cleaned_churn_risk_score > 0.8 THEN cleaned_monthly_revenue * 3  -- 3 months
            WHEN cleaned_churn_risk_score > 0.5 THEN cleaned_monthly_revenue * 12 -- 1 year
            WHEN cleaned_churn_risk_score > 0.3 THEN cleaned_monthly_revenue * 24 -- 2 years
            ELSE cleaned_monthly_revenue * 36 -- 3 years
        END as estimated_ltv
        
    FROM cleaned_customer_data
)

SELECT 
    *,
    -- Revenue growth calculation
    CASE 
        WHEN previous_revenue IS NULL OR previous_revenue = 0 THEN NULL
        ELSE ROUND(100.0 * (cleaned_monthly_revenue - previous_revenue) / previous_revenue, 2)
    END as revenue_growth_pct,
    
    -- Relative performance indicators
    CASE 
        WHEN revenue_percentile >= 0.9 THEN 'Top 10%'
        WHEN revenue_percentile >= 0.75 THEN 'Top 25%'
        WHEN revenue_percentile >= 0.5 THEN 'Above Average'
        ELSE 'Below Average'
    END as revenue_performance,
    
    CASE 
        WHEN health_percentile >= 0.9 THEN 'Excellent Health'
        WHEN health_percentile >= 0.75 THEN 'Good Health'
        WHEN health_percentile >= 0.5 THEN 'Average Health'
        ELSE 'Poor Health'
    END as health_performance

FROM customer_journey
ORDER BY signup_date, customer_id;

-- =====================================================
-- 3. BUSINESS INTELLIGENCE AGGREGATIONS
-- =====================================================

-- Create comprehensive business metrics summary
WITH business_metrics AS (
    SELECT 
        standardized_subscription_tier,
        standardized_industry,
        customer_lifecycle_stage,
        
        -- Customer counts
        COUNT(*) as customer_count,
        COUNT(CASE WHEN activation_completed = 1 THEN 1 END) as activated_customers,
        
        -- Revenue metrics
        SUM(cleaned_monthly_revenue) as total_monthly_revenue,
        AVG(cleaned_monthly_revenue) as avg_monthly_revenue,
        MEDIAN(cleaned_monthly_revenue) as median_monthly_revenue,
        
        -- Engagement metrics
        AVG(total_sessions) as avg_sessions,
        AVG(avg_session_duration) as avg_session_duration,
        AVG(cleaned_feature_usage_score) as avg_feature_usage,
        
        -- Health metrics
        AVG(cleaned_satisfaction_score) as avg_satisfaction,
        AVG(cleaned_churn_risk_score) as avg_churn_risk,
        
        -- Conversion metrics
        ROUND(100.0 * COUNT(CASE WHEN activation_completed = 1 THEN 1 END) / COUNT(*), 2) as activation_rate,
        
        -- Risk assessment
        COUNT(CASE WHEN cleaned_churn_risk_score > 0.7 THEN 1 END) as high_risk_customers,
        ROUND(100.0 * COUNT(CASE WHEN cleaned_churn_risk_score > 0.7 THEN 1 END) / COUNT(*), 2) as high_risk_rate
        
    FROM cleaned_customer_data
    GROUP BY GROUPING SETS (
        (standardized_subscription_tier),
        (standardized_industry),
        (customer_lifecycle_stage),
        (standardized_subscription_tier, standardized_industry),
        ()  -- Grand total
    )
)

SELECT 
    COALESCE(standardized_subscription_tier, 'ALL TIERS') as subscription_tier,
    COALESCE(standardized_industry, 'ALL INDUSTRIES') as industry,
    COALESCE(customer_lifecycle_stage, 'ALL STAGES') as lifecycle_stage,
    customer_count,
    activated_customers,
    ROUND(total_monthly_revenue, 2) as total_monthly_revenue,
    ROUND(avg_monthly_revenue, 2) as avg_monthly_revenue,
    ROUND(median_monthly_revenue, 2) as median_monthly_revenue,
    ROUND(avg_sessions, 1) as avg_sessions,
    ROUND(avg_session_duration, 1) as avg_session_duration,
    ROUND(avg_feature_usage, 3) as avg_feature_usage,
    ROUND(avg_satisfaction, 2) as avg_satisfaction,
    ROUND(avg_churn_risk, 3) as avg_churn_risk,
    activation_rate,
    high_risk_customers,
    high_risk_rate
FROM business_metrics
ORDER BY 
    CASE WHEN standardized_subscription_tier IS NULL THEN 1 ELSE 0 END,
    standardized_subscription_tier,
    CASE WHEN standardized_industry IS NULL THEN 1 ELSE 0 END,
    standardized_industry,
    customer_count DESC;
