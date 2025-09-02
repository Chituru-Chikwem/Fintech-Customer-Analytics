"""
Advanced Customer Segmentation using Multiple Clustering Algorithms
Demonstrates enterprise-level unsupervised learning with proper validation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedCustomerSegmentation:
    def __init__(self, data_path):
        """Initialize with customer data"""
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.models = {}
        self.results = {}
        
    def feature_engineering(self):
        """Advanced feature engineering for segmentation"""
        print("=== FEATURE ENGINEERING FOR SEGMENTATION ===")
        
        # Create composite features
        self.df['revenue_per_session'] = self.df['monthly_revenue'] / (self.df['total_sessions'] + 1)
        self.df['api_efficiency'] = self.df['api_calls_per_month'] / (self.df['total_sessions'] + 1)
        self.df['engagement_score'] = (
            self.df['feature_usage_score'] * 0.3 +
            self.df['avg_session_duration'] / 30 * 0.3 +  # Normalize to 0-1 scale
            (1 - self.df['churn_risk_score']) * 0.4  # Invert churn risk
        )
        
        # Customer lifecycle stage
        self.df['days_since_signup'] = (pd.to_datetime('2024-01-01') - pd.to_datetime(self.df['signup_date'])).dt.days
        
        def lifecycle_stage(days):
            if days <= 30:
                return 'New'
            elif days <= 90:
                return 'Growing'
            elif days <= 180:
                return 'Mature'
            else:
                return 'Veteran'
        
        self.df['lifecycle_stage'] = self.df['days_since_signup'].apply(lifecycle_stage)
        
        # Select features for clustering
        self.clustering_features = [
            'monthly_revenue', 'transaction_volume', 'feature_usage_score',
            'total_sessions', 'avg_session_duration', 'api_calls_per_month',
            'customer_satisfaction_score', 'company_size', 'annual_contract_value',
            'revenue_per_session', 'api_efficiency', 'engagement_score',
            'days_since_signup'
        ]
        
        # Handle missing values
        self.df[self.clustering_features] = self.df[self.clustering_features].fillna(
            self.df[self.clustering_features].median()
        )
        
        print(f"Created {len(self.clustering_features)} features for clustering")
        print(f"Feature correlation with revenue (top 5):")
        
        correlations = self.df[self.clustering_features + ['monthly_revenue']].corr()['monthly_revenue'].abs().sort_values(ascending=False)
        for feature, corr in correlations[1:6].items():  # Skip self-correlation
            print(f"  {feature}: {corr:.3f}")
    
    def optimal_clusters_analysis(self):
        """Determine optimal number of clusters using multiple methods"""
        print("\n=== OPTIMAL CLUSTERS ANALYSIS ===")
        
        # Prepare data
        X = self.df[self.clustering_features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Range of clusters to test
        k_range = range(2, 11)
        
        # Initialize metrics storage
        metrics = {
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        # Test different numbers of clusters
        for k in k_range:
            # K-Means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(X_scaled, cluster_labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, cluster_labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, cluster_labels))
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Elbow method
        axes[0,0].plot(k_range, metrics['inertia'], 'bo-')
        axes[0,0].set_title('Elbow Method (Inertia)')
        axes[0,0].set_xlabel('Number of Clusters')
        axes[0,0].set_ylabel('Inertia')
        axes[0,0].grid(True)
        
        # Silhouette score
        axes[0,1].plot(k_range, metrics['silhouette'], 'ro-')
        axes[0,1].set_title('Silhouette Score')
        axes[0,1].set_xlabel('Number of Clusters')
        axes[0,1].set_ylabel('Silhouette Score')
        axes[0,1].grid(True)
        
        # Calinski-Harabasz score
        axes[1,0].plot(k_range, metrics['calinski_harabasz'], 'go-')
        axes[1,0].set_title('Calinski-Harabasz Score')
        axes[1,0].set_xlabel('Number of Clusters')
        axes[1,0].set_ylabel('CH Score')
        axes[1,0].grid(True)
        
        # Davies-Bouldin score (lower is better)
        axes[1,1].plot(k_range, metrics['davies_bouldin'], 'mo-')
        axes[1,1].set_title('Davies-Bouldin Score')
        axes[1,1].set_xlabel('Number of Clusters')
        axes[1,1].set_ylabel('DB Score')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k
        optimal_k_silhouette = k_range[np.argmax(metrics['silhouette'])]
        optimal_k_ch = k_range[np.argmax(metrics['calinski_harabasz'])]
        optimal_k_db = k_range[np.argmin(metrics['davies_bouldin'])]
        
        print(f"Optimal clusters by Silhouette Score: {optimal_k_silhouette}")
        print(f"Optimal clusters by Calinski-Harabasz: {optimal_k_ch}")
        print(f"Optimal clusters by Davies-Bouldin: {optimal_k_db}")
        
        # Use majority vote or silhouette score as primary
        self.optimal_k = optimal_k_silhouette
        print(f"Selected optimal clusters: {self.optimal_k}")
        
        return metrics
    
    def multiple_clustering_algorithms(self):
        """Apply multiple clustering algorithms and compare results"""
        print(f"\n=== MULTIPLE CLUSTERING ALGORITHMS ===")
        
        # Prepare data
        X = self.df[self.clustering_features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize algorithms
        algorithms = {
            'K-Means': KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10),
            'Hierarchical': AgglomerativeClustering(n_clusters=self.optimal_k),
            'Gaussian Mixture': GaussianMixture(n_components=self.optimal_k, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        # Apply algorithms and evaluate
        algorithm_results = {}
        
        for name, algorithm in algorithms.items():
            print(f"\nApplying {name}...")
            
            if name == 'Gaussian Mixture':
                cluster_labels = algorithm.fit_predict(X_scaled)
            else:
                cluster_labels = algorithm.fit_predict(X_scaled)
            
            # Calculate metrics (skip DBSCAN if too many noise points)
            n_clusters = len(np.unique(cluster_labels))
            n_noise = np.sum(cluster_labels == -1) if -1 in cluster_labels else 0
            
            if n_clusters > 1 and n_noise < len(cluster_labels) * 0.5:
                silhouette = silhouette_score(X_scaled, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
                
                algorithm_results[name] = {
                    'labels': cluster_labels,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski_harabasz,
                    'davies_bouldin': davies_bouldin
                }
                
                print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")
                print(f"  Silhouette: {silhouette:.3f}")
                print(f"  Calinski-Harabasz: {calinski_harabasz:.1f}")
                print(f"  Davies-Bouldin: {davies_bouldin:.3f}")
            else:
                print(f"  Skipped - too many noise points or insufficient clusters")
        
        # Select best algorithm based on silhouette score
        best_algorithm = max(algorithm_results.keys(), 
                           key=lambda x: algorithm_results[x]['silhouette'])
        
        print(f"\nBest algorithm: {best_algorithm}")
        
        # Store best results
        self.best_labels = algorithm_results[best_algorithm]['labels']
        self.df['cluster'] = self.best_labels
        
        self.results['clustering_comparison'] = algorithm_results
        self.results['best_algorithm'] = best_algorithm
        
        return algorithm_results
    
    def cluster_profiling(self):
        """Comprehensive cluster profiling and business interpretation"""
        print(f"\n=== CLUSTER PROFILING ===")
        
        # Calculate cluster statistics
        cluster_profiles = self.df.groupby('cluster').agg({
            'monthly_revenue': ['mean', 'median', 'std', 'count'],
            'company_size': 'mean',
            'feature_usage_score': 'mean',
            'churn_risk_score': 'mean',
            'customer_satisfaction_score': 'mean',
            'total_sessions': 'mean',
            'annual_contract_value': 'mean',
            'engagement_score': 'mean'
        }).round(2)
        
        print("Cluster Profiles:")
        print(cluster_profiles)
        
        # Business interpretation
        cluster_names = {}
        cluster_insights = {}
        
        for cluster_id in sorted(self.df['cluster'].unique()):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            avg_revenue = cluster_data['monthly_revenue'].mean()
            avg_churn_risk = cluster_data['churn_risk_score'].mean()
            avg_satisfaction = cluster_data['customer_satisfaction_score'].mean()
            avg_engagement = cluster_data['engagement_score'].mean()
            
            # Name clusters based on characteristics
            if avg_revenue > self.df['monthly_revenue'].quantile(0.75):
                if avg_churn_risk < 0.3:
                    name = "High-Value Champions"
                else:
                    name = "High-Value At-Risk"
            elif avg_revenue > self.df['monthly_revenue'].median():
                if avg_engagement > 0.6:
                    name = "Growing Advocates"
                else:
                    name = "Moderate Users"
            else:
                if avg_churn_risk > 0.6:
                    name = "At-Risk Low-Value"
                else:
                    name = "New/Small Customers"
            
            cluster_names[cluster_id] = name
            cluster_insights[cluster_id] = {
                'name': name,
                'size': len(cluster_data),
                'avg_revenue': avg_revenue,
                'revenue_share': (cluster_data['monthly_revenue'].sum() / self.df['monthly_revenue'].sum()) * 100,
                'avg_churn_risk': avg_churn_risk,
                'avg_satisfaction': avg_satisfaction,
                'avg_engagement': avg_engagement
            }
        
        # Print business insights
        print(f"\nBusiness Cluster Insights:")
        for cluster_id, insights in cluster_insights.items():
            print(f"\nCluster {cluster_id}: {insights['name']}")
            print(f"  Size: {insights['size']} customers ({insights['size']/len(self.df):.1%})")
            print(f"  Avg Revenue: ${insights['avg_revenue']:,.2f}")
            print(f"  Revenue Share: {insights['revenue_share']:.1f}%")
            print(f"  Churn Risk: {insights['avg_churn_risk']:.3f}")
            print(f"  Satisfaction: {insights['avg_satisfaction']:.2f}/5.0")
            print(f"  Engagement: {insights['avg_engagement']:.3f}")
        
        # Add cluster names to dataframe
        self.df['cluster_name'] = self.df['cluster'].map(cluster_names)
        
        self.results['cluster_profiles'] = cluster_profiles
        self.results['cluster_insights'] = cluster_insights
        
        return cluster_insights
    
    def advanced_visualizations(self):
        """Create advanced cluster visualizations"""
        print(f"\n=== ADVANCED CLUSTER VISUALIZATIONS ===")
        
        # PCA for 2D visualization
        X = self.df[self.clustering_features]
        X_scaled = self.scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization dataframe
        viz_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'cluster': self.df['cluster'],
            'cluster_name': self.df['cluster_name'],
            'monthly_revenue': self.df['monthly_revenue'],
            'company_size': self.df['company_size'],
            'churn_risk': self.df['churn_risk_score']
        })
        
        # PCA scatter plot
        fig = px.scatter(viz_df, x='PC1', y='PC2', color='cluster_name', 
                        size='monthly_revenue', hover_data=['company_size', 'churn_risk'],
                        title=f'Customer Segments (PCA Visualization)<br>Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}')
        fig.show()
        
        # t-SNE for non-linear visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        viz_df_tsne = pd.DataFrame({
            'TSNE1': X_tsne[:, 0],
            'TSNE2': X_tsne[:, 1],
            'cluster_name': self.df['cluster_name'],
            'monthly_revenue': self.df['monthly_revenue']
        })
        
        fig_tsne = px.scatter(viz_df_tsne, x='TSNE1', y='TSNE2', color='cluster_name',
                             size='monthly_revenue', title='Customer Segments (t-SNE Visualization)')
        fig_tsne.show()
        
        # Cluster comparison radar chart
        cluster_metrics = self.df.groupby('cluster_name').agg({
            'monthly_revenue': lambda x: (x.mean() - self.df['monthly_revenue'].min()) / (self.df['monthly_revenue'].max() - self.df['monthly_revenue'].min()),
            'feature_usage_score': 'mean',
            'customer_satisfaction_score': lambda x: x.mean() / 5,  # Normalize to 0-1
            'engagement_score': 'mean',
            'churn_risk_score': lambda x: 1 - x.mean()  # Invert for radar chart
        }).round(3)
        
        fig_radar = go.Figure()
        
        for cluster_name in cluster_metrics.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=cluster_metrics.loc[cluster_name].values,
                theta=['Revenue', 'Feature Usage', 'Satisfaction', 'Engagement', 'Retention'],
                fill='toself',
                name=cluster_name
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Customer Segment Comparison (Radar Chart)"
        )
        fig_radar.show()
    
    def generate_segmentation_report(self):
        """Generate comprehensive segmentation report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE CUSTOMER SEGMENTATION REPORT")
        print("="*60)
        
        # Run complete analysis
        self.feature_engineering()
        metrics = self.optimal_clusters_analysis()
        algorithm_results = self.multiple_clustering_algorithms()
        cluster_insights = self.cluster_profiling()
        self.advanced_visualizations()
        
        # Summary statistics
        print(f"\n=== SEGMENTATION SUMMARY ===")
        print(f"Optimal number of clusters: {self.optimal_k}")
        print(f"Best algorithm: {self.results['best_algorithm']}")
        print(f"Silhouette score: {self.results['clustering_comparison'][self.results['best_algorithm']]['silhouette']:.3f}")
        
        # Business impact
        total_revenue = self.df['monthly_revenue'].sum()
        high_value_clusters = [name for name, insights in cluster_insights.items() 
                              if 'High-Value' in insights['name'] or 'Champions' in insights['name']]
        
        if high_value_clusters:
            high_value_revenue = sum([insights['revenue_share'] for cluster_id, insights in cluster_insights.items() 
                                    if cluster_id in high_value_clusters])
            high_value_customers = sum([insights['size'] for cluster_id, insights in cluster_insights.items() 
                                      if cluster_id in high_value_clusters])
            
            print(f"\nHigh-Value Segments:")
            print(f"  Customers: {high_value_customers} ({high_value_customers/len(self.df):.1%})")
            print(f"  Revenue Share: {high_value_revenue:.1f}%")
        
        return self.results

if __name__ == "__main__":
    # Initialize segmentation analysis
    segmentation = AdvancedCustomerSegmentation('data/fintech_customer_data.csv')
    
    # Generate comprehensive report
    results = segmentation.generate_segmentation_report()
    
    print(f"\n[v0] Customer segmentation complete with {segmentation.optimal_k} segments identified.")
    print(f"[v0] Analysis includes multiple algorithms, statistical validation, and business insights.")
    print(f"[v0] Visualizations and cluster profiles ready for executive presentation.")
