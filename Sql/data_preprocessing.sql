-- Advanced Data Preprocessing and Cleaning
-- Demonstrates enterprise-level SQL data quality and preprocessing techniques

-- =====================================================
-- 1. DATA QUALITY ASSESSMENT AND CLEANING
-- =====================================================

-- Create a comprehensive data quality report
WITH data_quality_metrics AS (
    SELECT 
        'customer_id' as column_name,
        COUNT(*) as total_records,
        COUNT(customer_id) as non_null_count,
        COUNT(DISTINCT customer_id) as unique_count,
        ROUND(100.0 * COUNT(customer_id) / COUNT(*), 2) as completeness_pct,
        CASE WHEN COUNT(*) = COUNT(DISTINCT customer_id) THEN 'PASS' ELSE 'FAIL' END as uniqueness_check
    FROM fintech_customers
    
    UNION ALL
    
    SELECT 
        'monthly_revenue',
        COUNT(*),
        COUNT(monthly_revenue),
        COUNT(DISTINCT monthly_revenue),
        ROUND(100.0 * COUNT(monthly_revenue) / COUNT(*), 2),
        CASE WHEN MIN(monthly_revenue) >= 0 THEN 'PASS' ELSE 'FAIL' END
    FROM fintech_customers
    
    UNION ALL
    
    SELECT 
        'churn_risk_score',
        COUNT(*),
        COUNT(churn_risk_score),
        COUNT(DISTINCT churn_risk_score),
        ROUND(100.0 * COUNT(churn_risk_score) / COUNT(*), 2),
        CASE WHEN MIN(churn_risk_score) >= 0 AND MAX(churn_risk_score) <= 1 THEN 'PASS' ELSE 'FAIL' END
    FROM fintech_customers
    
    UNION ALL
    
    SELECT 
        'customer_satisfaction_score',
        COUNT(*),
        COUNT(customer_satisfaction_score),
        COUNT(DISTINCT customer_satisfaction_score),
        ROUND(100.0 * COUNT(customer_satisfaction_score) / COUNT(*), 2),
        CASE WHEN MIN(customer_satisfaction_score) >= 1 AND MAX(customer_satisfaction_score) <= 5 THEN 'PASS' ELSE 'FAIL' END
    FROM fintech_customers
)
SELECT * FROM data_quality_metrics
ORDER BY completeness_pct DESC;

-- =====================================================
-- 2. OUTLIER DETECTION AND HANDLING
-- =====================================================

-- Identify outliers using IQR method for key numeric columns
WITH revenue_quartiles AS (
    SELECT 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY monthly_revenue) as q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY monthly_revenue) as q3,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY monthly_revenue) - 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY monthly_revenue) as iqr
    FROM fintech_customers
),
outlier_analysis AS (
    SELECT 
        c.*,
        q.q1,
        q.q3,
        q.iqr,
        q.q1 - 1.5 * q.iqr as lower_bound,
        q.q3 + 1.5 * q.iqr as upper_bound,
        CASE 
            WHEN c.monthly_revenue < (q.q1 - 1.5 * q.iqr) OR 
                 c.monthly_revenue > (q.q3 + 1.5 * q.iqr) 
            THEN 'OUTLIER' 
            ELSE 'NORMAL' 
        END as revenue_outlier_flag
    FROM fintech_customers c
    CROSS JOIN revenue_quartiles q
)
SELECT 
    revenue_outlier_flag,
    COUNT(*) as customer_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as percentage,
    ROUND(AVG(monthly_revenue), 2) as avg_revenue,
    ROUND(MIN(monthly_revenue), 2) as min_revenue,
    ROUND(MAX(monthly_revenue), 2) as max_revenue
FROM outlier_analysis
GROUP BY revenue_outlier_flag
ORDER BY revenue_outlier_flag;

-- =====================================================
-- 3. ADVANCED DATA CLEANING AND STANDARDIZATION
-- =====================================================

-- Create a cleaned and standardized dataset
CREATE OR REPLACE VIEW cleaned_customer_data AS
WITH data_cleaning AS (
    SELECT 
        customer_id,
        signup_date,
        
        -- Standardize and validate age
        CASE 
            WHEN age BETWEEN 18 AND 80 THEN age
            WHEN age < 18 THEN 25  -- Default for invalid ages
            WHEN age > 80 THEN 65  -- Cap extreme ages
            ELSE 35  -- Default for nulls
        END as cleaned_age,
        
        -- Standardize company size categories
        CASE 
            WHEN company_size <= 10 THEN 'Micro (1-10)'
            WHEN company_size <= 50 THEN 'Small (11-50)'
            WHEN company_size <= 200 THEN 'Medium (51-200)'
            WHEN company_size <= 1000 THEN 'Large (201-1000)'
            ELSE 'Enterprise (1000+)'
        END as company_size_category,
        
        -- Clean and standardize industry names
        CASE 
            WHEN LOWER(industry) LIKE '%tech%' THEN 'Technology'
            WHEN LOWER(industry) LIKE '%health%' THEN 'Healthcare'
            WHEN LOWER(industry) LIKE '%financ%' OR LOWER(industry) LIKE '%bank%' THEN 'Finance'
            WHEN LOWER(industry) LIKE '%retail%' OR LOWER(industry) LIKE '%ecommerce%' THEN 'Retail'
            WHEN LOWER(industry) LIKE '%manufact%' THEN 'Manufacturing'
            ELSE INITCAP(industry)
        END as standardized_industry,
        
        -- Handle missing and extreme values for revenue
        CASE 
            WHEN monthly_revenue IS NULL THEN 0
            WHEN monthly_revenue < 0 THEN 0
            WHEN monthly_revenue > 1000000 THEN 1000000  -- Cap extreme values
            ELSE monthly_revenue
        END as cleaned_monthly_revenue,
        
        -- Validate and clean scores (0-1 range)
        GREATEST(0, LEAST(1, COALESCE(feature_usage_score, 0.5))) as cleaned_feature_usage_score,
        GREATEST(0, LEAST(1, COALESCE(churn_risk_score, 0.5))) as cleaned_churn_risk_score,
        
        -- Validate satisfaction score (1-5 range)
        GREATEST(1, LEAST(5, COALESCE(customer_satisfaction_score, 3))) as cleaned_satisfaction_score,
        
        -- Clean boolean flags
        COALESCE(activation_completed, 0) as activation_completed,
        
        -- Standardize subscription tiers
        CASE 
            WHEN UPPER(subscription_tier) IN ('BASIC', 'STARTER', 'FREE') THEN 'Basic'
            WHEN UPPER(subscription_tier) IN ('PREMIUM', 'PRO', 'PROFESSIONAL') THEN 'Premium'
            WHEN UPPER(subscription_tier) IN ('ENTERPRISE', 'BUSINESS', 'CORPORATE') THEN 'Enterprise'
            ELSE 'Basic'
        END as standardized_subscription_tier,
        
        -- Original columns for comparison
        company_size,
        industry,
        monthly_revenue,
        feature_usage_score,
        churn_risk_score,
        customer_satisfaction_score,
        subscription_tier,
        
        -- Additional columns
        transaction_volume,
        support_tickets,
        days_since_last_login,
        total_sessions,
        avg_session_duration,
        api_calls_per_month,
        integration_count,
        team_size,
        annual_contract_value,
        payment_method,
        referral_source,
        onboarding_completion_rate,
        feature_adoption_score,
        location
        
    FROM fintech_customers
)
SELECT 
    *,
    -- Add data quality flags
    CASE 
        WHEN cleaned_age != age THEN 1 ELSE 0 
    END as age_was_cleaned,
    
    CASE 
        WHEN cleaned_monthly_revenue != monthly_revenue THEN 1 ELSE 0 
    END as revenue_was_cleaned,
    
    CASE 
        WHEN standardized_industry != industry THEN 1 ELSE 0 
    END as industry_was_standardized,
    
    -- Calculate days since signup
    CURRENT_DATE - signup_date as days_since_signup,
    
    -- Create customer lifecycle stage
    CASE 
        WHEN CURRENT_DATE - signup_date <= 30 THEN 'New (0-30 days)'
        WHEN CURRENT_DATE - signup_date <= 90 THEN 'Growing (31-90 days)'
        WHEN CURRENT_DATE - signup_date <= 180 THEN 'Mature (91-180 days)'
        WHEN CURRENT_DATE - signup_date <= 365 THEN 'Established (181-365 days)'
        ELSE 'Veteran (365+ days)'
    END as customer_lifecycle_stage

FROM data_cleaning;

-- =====================================================
-- 4. DATA VALIDATION SUMMARY
-- =====================================================

-- Generate comprehensive data validation report
SELECT 
    'Data Cleaning Summary' as report_section,
    COUNT(*) as total_records,
    SUM(age_was_cleaned) as age_corrections,
    SUM(revenue_was_cleaned) as revenue_corrections,
    SUM(industry_was_standardized) as industry_standardizations,
    ROUND(100.0 * SUM(age_was_cleaned) / COUNT(*), 2) as age_correction_rate,
    ROUND(100.0 * SUM(revenue_was_cleaned) / COUNT(*), 2) as revenue_correction_rate,
    ROUND(100.0 * SUM(industry_was_standardized) / COUNT(*), 2) as industry_standardization_rate
FROM cleaned_customer_data

UNION ALL

SELECT 
    'Lifecycle Distribution',
    COUNT(*),
    NULL, NULL, NULL,
    NULL, NULL, NULL
FROM cleaned_customer_data

UNION ALL

SELECT 
    customer_lifecycle_stage,
    COUNT(*),
    NULL, NULL, NULL,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2),
    NULL, NULL
FROM cleaned_customer_data
GROUP BY customer_lifecycle_stage
ORDER BY report_section, total_records DESC;

-- =====================================================
-- 5. CREATE INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Create indexes on frequently queried columns
-- Note: These would be actual CREATE INDEX statements in a real database

/*
CREATE INDEX idx_customers_signup_date ON fintech_customers(signup_date);
CREATE INDEX idx_customers_subscription_tier ON fintech_customers(subscription_tier);
CREATE INDEX idx_customers_industry ON fintech_customers(industry);
CREATE INDEX idx_customers_monthly_revenue ON fintech_customers(monthly_revenue);
CREATE INDEX idx_customers_churn_risk ON fintech_customers(churn_risk_score);
CREATE INDEX idx_customers_activation ON fintech_customers(activation_completed);

-- Composite indexes for common query patterns
CREATE INDEX idx_customers_tier_industry ON fintech_customers(subscription_tier, industry);
CREATE INDEX idx_customers_revenue_churn ON fintech_customers(monthly_revenue, churn_risk_score);
CREATE INDEX idx_customers_signup_activation ON fintech_customers(signup_date, activation_completed);
*/

-- Performance analysis query
SELECT 
    'Index Recommendations' as analysis_type,
    'High cardinality columns need indexes' as recommendation,
    COUNT(DISTINCT customer_id) as unique_customer_ids,
    COUNT(DISTINCT industry) as unique_industries,
    COUNT(DISTINCT subscription_tier) as unique_tiers,
    COUNT(DISTINCT location) as unique_locations
FROM fintech_customers;
