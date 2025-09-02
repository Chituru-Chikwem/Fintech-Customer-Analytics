"""
Comprehensive Exploratory Data Analysis (EDA)
Advanced statistical exploration with visualization and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEDA:
    def __init__(self, data_path):
        """Initialize EDA with customer data"""
        self.df = pd.read_csv(data_path)
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Configure professional plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def univariate_analysis(self):
        """Comprehensive univariate analysis of all variables"""
        print("=== UNIVARIATE ANALYSIS ===")
        
        # Numeric variables analysis
        numeric_vars = self.df.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, var in enumerate(numeric_vars[:16]):  # Top 16 numeric variables
            # Distribution plot with statistical annotations
            sns.histplot(data=self.df, x=var, kde=True, ax=axes[i])
            
            # Add statistical information
            mean_val = self.df[var].mean()
            median_val = self.df[var].median()
            std_val = self.df[var].std()
            skewness = stats.skew(self.df[var].dropna())
            
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
            axes[i].set_title(f'{var}\nSkewness: {skewness:.2f}, Std: {std_val:.2f}')
            axes[i].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('univariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Categorical variables analysis
        categorical_vars = ['subscription_tier', 'industry', 'location', 'referral_source']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, var in enumerate(categorical_vars):
            value_counts = self.df[var].value_counts()
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i])
            axes[i].set_title(f'{var} Distribution')
            
            # Add percentage labels
            total = len(self.df)
            for j, v in enumerate(value_counts.values):
                axes[i].text(v + 0.1, j, f'{v/total:.1%}', va='center')
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def bivariate_analysis(self):
        """Advanced bivariate analysis with statistical testing"""
        print("\n=== BIVARIATE ANALYSIS ===")
        
        # Revenue analysis by subscription tier
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot with statistical annotations
        sns.boxplot(data=self.df, x='subscription_tier', y='monthly_revenue', ax=axes[0,0])
        axes[0,0].set_title('Monthly Revenue by Subscription Tier')
        
        # Add statistical test results
        basic_rev = self.df[self.df['subscription_tier'] == 'Basic']['monthly_revenue']
        premium_rev = self.df[self.df['subscription_tier'] == 'Premium']['monthly_revenue']
        enterprise_rev = self.df[self.df['subscription_tier'] == 'Enterprise']['monthly_revenue']
        
        # ANOVA test
        f_stat, p_value = stats.f_oneway(basic_rev.dropna(), premium_rev.dropna(), enterprise_rev.dropna())
        axes[0,0].text(0.02, 0.98, f'ANOVA: F={f_stat:.2f}, p={p_value:.4f}', 
                      transform=axes[0,0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Churn risk by industry
        sns.violinplot(data=self.df, x='industry', y='churn_risk_score', ax=axes[0,1])
        axes[0,1].set_title('Churn Risk Distribution by Industry')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Feature usage vs satisfaction scatter
        sns.scatterplot(data=self.df, x='feature_usage_score', y='customer_satisfaction_score', 
                       hue='subscription_tier', size='monthly_revenue', ax=axes[1,0])
        axes[1,0].set_title('Feature Usage vs Customer Satisfaction')
        
        # Correlation with revenue
        correlation_r, correlation_p = stats.pearsonr(self.df['feature_usage_score'].dropna(), 
                                                     self.df['customer_satisfaction_score'].dropna())
        axes[1,0].text(0.02, 0.98, f'r={correlation_r:.3f}, p={correlation_p:.4f}', 
                      transform=axes[1,0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Activation completion by referral source
        activation_by_source = pd.crosstab(self.df['referral_source'], self.df['activation_completed'], normalize='index')
        activation_by_source.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Activation Rate by Referral Source')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(['Not Activated', 'Activated'])
        
        plt.tight_layout()
        plt.savefig('bivariate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def correlation_heatmap(self):
        """Advanced correlation analysis with clustering"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Select numeric variables for correlation
        numeric_vars = ['monthly_revenue', 'transaction_volume', 'feature_usage_score', 
                       'churn_risk_score', 'total_sessions', 'avg_session_duration',
                       'api_calls_per_month', 'customer_satisfaction_score', 'age',
                       'company_size', 'annual_contract_value']
        
        correlation_matrix = self.df[numeric_vars].corr()
        
        # Create clustered heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Customer Metrics Correlation Matrix\n(Lower Triangle Only)', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identify strongest correlations
        correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                correlation_pairs.append({
                    'var1': correlation_matrix.columns[i],
                    'var2': correlation_matrix.columns[j],
                    'correlation': correlation_matrix.iloc[i, j]
                })
        
        # Sort by absolute correlation
        correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print("Top 10 Strongest Correlations:")
        for i, pair in enumerate(correlation_pairs[:10]):
            print(f"{i+1:2d}. {pair['var1']} vs {pair['var2']}: {pair['correlation']:.3f}")
    
    def advanced_visualizations(self):
        """Create advanced interactive visualizations"""
        print("\n=== ADVANCED VISUALIZATIONS ===")
        
        # 3D scatter plot of key metrics
        fig = px.scatter_3d(self.df, x='monthly_revenue', y='feature_usage_score', z='customer_satisfaction_score',
                           color='subscription_tier', size='company_size', hover_data=['industry'],
                           title='3D Customer Analysis: Revenue vs Usage vs Satisfaction')
        fig.show()
        
        # Customer journey funnel
        funnel_data = self.df.groupby('subscription_tier').agg({
            'activation_completed': 'mean',
            'feature_adoption_score': 'mean',
            'customer_satisfaction_score': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        for tier in funnel_data['subscription_tier']:
            tier_data = funnel_data[funnel_data['subscription_tier'] == tier]
            fig.add_trace(go.Scatter(
                x=['Activation', 'Feature Adoption', 'Satisfaction'],
                y=[tier_data['activation_completed'].iloc[0], 
                   tier_data['feature_adoption_score'].iloc[0],
                   tier_data['customer_satisfaction_score'].iloc[0]/5],  # Normalize to 0-1
                mode='lines+markers',
                name=tier,
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig.update_layout(title='Customer Journey by Subscription Tier',
                         xaxis_title='Journey Stage',
                         yaxis_title='Score (0-1 scale)',
                         height=500)
        fig.show()
        
        # Revenue distribution by multiple dimensions
        fig = px.sunburst(self.df, path=['industry', 'subscription_tier', 'referral_source'], 
                         values='monthly_revenue',
                         title='Revenue Distribution: Industry → Tier → Source')
        fig.show()
    
    def statistical_summary_report(self):
        """Generate comprehensive statistical summary"""
        print("\n" + "="*60)
        print("COMPREHENSIVE EDA STATISTICAL REPORT")
        print("="*60)
        
        # Dataset overview
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Duplicate Rows: {self.df.duplicated().sum()}")
        
        # Numeric variables summary
        numeric_summary = self.df.describe()
        print(f"\nNumeric Variables Summary:")
        print(numeric_summary.round(2))
        
        # Categorical variables summary
        categorical_vars = self.df.select_dtypes(include=['object']).columns
        print(f"\nCategorical Variables Summary:")
        for var in categorical_vars:
            unique_count = self.df[var].nunique()
            most_common = self.df[var].mode()[0]
            print(f"{var}: {unique_count} unique values, most common: {most_common}")
        
        # Data quality metrics
        print(f"\nData Quality Metrics:")
        print(f"Missing Values: {self.df.isnull().sum().sum()} ({self.df.isnull().sum().sum()/self.df.size:.1%})")
        
        # Business metrics
        print(f"\nKey Business Metrics:")
        print(f"Total Monthly Revenue: ${self.df['monthly_revenue'].sum():,.2f}")
        print(f"Average Customer Value: ${self.df['monthly_revenue'].mean():,.2f}")
        print(f"Activation Rate: {self.df['activation_completed'].mean():.1%}")
        print(f"Average Churn Risk: {self.df['churn_risk_score'].mean():.3f}")
        print(f"Customer Satisfaction: {self.df['customer_satisfaction_score'].mean():.2f}/5.0")
        
    def run_complete_eda(self):
        """Execute complete EDA pipeline"""
        print("Starting Comprehensive Exploratory Data Analysis...")
        
        self.statistical_summary_report()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.correlation_heatmap()
        self.advanced_visualizations()
        
        print(f"\n[v0] EDA complete. Generated professional visualizations and statistical insights.")
        print(f"[v0] Analysis includes distribution analysis, correlation testing, and advanced visualizations.")
        print(f"[v0] All plots saved as high-resolution PNG files for portfolio inclusion.")

if __name__ == "__main__":
    # Initialize EDA
    eda = ComprehensiveEDA('data/fintech_customer_data.csv')
    
    # Run complete analysis
    eda.run_complete_eda()
