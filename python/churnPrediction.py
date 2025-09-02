"""
Advanced Churn Prediction with Multiple ML Models
Enterprise-level classification with proper validation and interpretability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings('ignore')

class AdvancedChurnPrediction:
    def __init__(self, data_path):
        """Initialize with customer data"""
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def feature_engineering_for_churn(self):
        """Advanced feature engineering for churn prediction"""
        print("=== FEATURE ENGINEERING FOR CHURN PREDICTION ===")
        
        # Create churn target variable (high churn risk = 1, low = 0)
        self.df['churn_target'] = (self.df['churn_risk_score'] > 0.5).astype(int)
        
        # Behavioral features
        self.df['revenue_per_session'] = self.df['monthly_revenue'] / (self.df['total_sessions'] + 1)
        self.df['api_efficiency'] = self.df['api_calls_per_month'] / (self.df['total_sessions'] + 1)
        self.df['support_intensity'] = self.df['support_tickets'] / (self.df['days_since_last_login'] + 1)
        
        # Engagement metrics
        self.df['low_engagement'] = (self.df['feature_usage_score'] < 0.3).astype(int)
        self.df['high_session_duration'] = (self.df['avg_session_duration'] > self.df['avg_session_duration'].median()).astype(int)
        self.df['frequent_user'] = (self.df['total_sessions'] > self.df['total_sessions'].quantile(0.75)).astype(int)
        
        # Customer lifecycle
        self.df['days_since_signup'] = (pd.to_datetime('2024-01-01') - pd.to_datetime(self.df['signup_date'])).dt.days
        self.df['new_customer'] = (self.df['days_since_signup'] <= 30).astype(int)
        self.df['veteran_customer'] = (self.df['days_since_signup'] > 180).astype(int)
        
        # Business value indicators
        self.df['high_value'] = (self.df['monthly_revenue'] > self.df['monthly_revenue'].quantile(0.75)).astype(int)
        self.df['enterprise_customer'] = (self.df['subscription_tier'] == 'Enterprise').astype(int)
        
        # Encode categorical variables
        le_industry = LabelEncoder()
        le_location = LabelEncoder()
        le_referral = LabelEncoder()
        le_payment = LabelEncoder()
        
        self.df['industry_encoded'] = le_industry.fit_transform(self.df['industry'])
        self.df['location_encoded'] = le_location.fit_transform(self.df['location'])
        self.df['referral_encoded'] = le_referral.fit_transform(self.df['referral_source'])
        self.df['payment_encoded'] = le_payment.fit_transform(self.df['payment_method'])
        
        # Select features for modeling
        self.feature_columns = [
            'monthly_revenue', 'transaction_volume', 'feature_usage_score',
            'support_tickets', 'days_since_last_login', 'total_sessions',
            'avg_session_duration', 'api_calls_per_month', 'integration_count',
            'team_size', 'annual_contract_value', 'customer_satisfaction_score',
            'company_size', 'age', 'onboarding_completion_rate', 'feature_adoption_score',
            'revenue_per_session', 'api_efficiency', 'support_intensity',
            'low_engagement', 'high_session_duration', 'frequent_user',
            'new_customer', 'veteran_customer', 'high_value', 'enterprise_customer',
            'industry_encoded', 'location_encoded', 'referral_encoded', 'payment_encoded'
        ]
        
        # Handle missing values
        self.df[self.feature_columns] = self.df[self.feature_columns].fillna(
            self.df[self.feature_columns].median()
        )
        
        print(f"Created {len(self.feature_columns)} features for churn prediction")
        print(f"Churn rate: {self.df['churn_target'].mean():.1%}")
        print(f"Class distribution: {self.df['churn_target'].value_counts().to_dict()}")
        
    def feature_selection(self):
        """Advanced feature selection using multiple methods"""
        print("\n=== FEATURE SELECTION ===")
        
        X = self.df[self.feature_columns]
        y = self.df['churn_target']
        
        # Method 1: Statistical feature selection
        selector_stats = SelectKBest(score_func=f_classif, k=15)
        X_selected_stats = selector_stats.fit_transform(X, y)
        selected_features_stats = [self.feature_columns[i] for i in selector_stats.get_support(indices=True)]
        
        # Method 2: Recursive Feature Elimination with Random Forest
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector_rfe = RFE(rf_selector, n_features_to_select=15)
        X_selected_rfe = selector_rfe.fit_transform(X, y)
        selected_features_rfe = [self.feature_columns[i] for i in selector_rfe.get_support(indices=True)]
        
        # Method 3: Feature importance from Random Forest
        rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_importance.fit(X, y)
        feature_importance_scores = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_importance.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features_importance = feature_importance_scores.head(15)['feature'].tolist()
        
        # Combine methods (union of top features)
        all_selected = set(selected_features_stats + selected_features_rfe + selected_features_importance)
        self.selected_features = list(all_selected)
        
        print(f"Statistical selection: {len(selected_features_stats)} features")
        print(f"RFE selection: {len(selected_features_rfe)} features")
        print(f"Importance selection: {len(selected_features_importance)} features")
        print(f"Combined selection: {len(self.selected_features)} features")
        
        # Display top features by importance
        print(f"\nTop 10 Features by Importance:")
        for i, (_, row) in enumerate(feature_importance_scores.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        self.results['feature_selection'] = {
            'selected_features': self.selected_features,
            'feature_importance': feature_importance_scores
        }
        
        return self.selected_features
    
    def model_training_and_validation(self):
        """Train multiple models with proper validation"""
        print("\n=== MODEL TRAINING AND VALIDATION ===")
        
        # Prepare data
        X = self.df[self.selected_features]
        y = self.df['churn_target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train and evaluate models
        model_results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=cv, scoring='roc_auc')
            
            # Train final model
            model.fit(X_train_model, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_model)
            y_pred_proba = model.predict_proba(X_test_model)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            model_results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_f1': f1,
                'test_roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test F1-Score: {f1:.4f}")
            print(f"  Test ROC-AUC: {roc_auc:.4f}")
        
        # Select best model based on CV ROC-AUC
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['cv_mean'])
        self.best_model = model_results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        
        # Store results
        self.X_test = X_test
        self.y_test = y_test
        self.model_results = model_results
        self.results['model_comparison'] = model_results
        
        return model_results
    
    def hyperparameter_optimization(self):
        """Optimize hyperparameters for the best model"""
        print(f"\n=== HYPERPARAMETER OPTIMIZATION FOR {self.best_model_name} ===")
        
        # Prepare data
        X = self.df[self.selected_features]
        y = self.df['churn_target']
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                self.best_model, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            
            # Use scaled data for Logistic Regression
            if self.best_model_name == 'Logistic Regression':
                X_scaled = self.scaler.fit_transform(X)
                grid_search.fit(X_scaled, y)
            else:
                grid_search.fit(X, y)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update best model
            self.best_model = grid_search.best_estimator_
            self.results['best_params'] = grid_search.best_params_
            self.results['best_cv_score'] = grid_search.best_score_
        
    def model_interpretability(self):
        """Advanced model interpretability analysis"""
        print(f"\n=== MODEL INTERPRETABILITY ===")
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(15), y='feature', x='importance')
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance_churn.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            self.feature_importance = feature_importance
        
        # SHAP analysis for model interpretability
        try:
            print("\nGenerating SHAP explanations...")
            
            # Prepare data for SHAP
            X_sample = self.X_test.sample(min(100, len(self.X_test)), random_state=42)
            
            if self.best_model_name == 'Random Forest':
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(X_sample)
                
                # Use class 1 (churn) SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.tight_layout()
                plt.savefig('shap_summary_churn.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print("SHAP analysis complete - visualizations saved")
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    
    def advanced_evaluation_metrics(self):
        """Comprehensive model evaluation with business metrics"""
        print(f"\n=== ADVANCED EVALUATION METRICS ===")
        
        # Get predictions from best model
        best_results = self.model_results[self.best_model_name]
        y_pred = best_results['y_pred']
        y_pred_proba = best_results['y_pred_proba']
        
        # Detailed classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix_churn.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curve_churn.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('precision_recall_curve_churn.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Business impact analysis
        print(f"\n=== BUSINESS IMPACT ANALYSIS ===")
        
        # Calculate potential savings from churn prevention
        high_risk_customers = self.df[self.df['churn_target'] == 1]
        potential_lost_revenue = high_risk_customers['monthly_revenue'].sum() * 12  # Annual
        
        # Assuming 30% of predicted churners can be saved with intervention
        intervention_success_rate = 0.3
        predicted_churners = len(y_pred[y_pred == 1])
        actual_churners = len(self.y_test[self.y_test == 1])
        
        # True positive rate (recall)
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        
        potential_savings = potential_lost_revenue * recall * intervention_success_rate
        
        print(f"Potential annual revenue at risk: ${potential_lost_revenue:,.2f}")
        print(f"Model recall (churn detection rate): {recall:.1%}")
        print(f"Potential annual savings (30% intervention success): ${potential_savings:,.2f}")
        
        self.results['business_impact'] = {
            'potential_lost_revenue': potential_lost_revenue,
            'recall': recall,
            'potential_savings': potential_savings
        }
    
    def generate_churn_prediction_report(self):
        """Generate comprehensive churn prediction report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE CHURN PREDICTION REPORT")
        print("="*60)
        
        # Run complete analysis
        self.feature_engineering_for_churn()
        self.feature_selection()
        self.model_training_and_validation()
        self.hyperparameter_optimization()
        self.model_interpretability()
        self.advanced_evaluation_metrics()
        
        # Summary
        best_results = self.model_results[self.best_model_name]
        print(f"\n=== MODEL PERFORMANCE SUMMARY ===")
        print(f"Best Model: {self.best_model_name}")
        print(f"Test Accuracy: {best_results['test_accuracy']:.4f}")
        print(f"Test F1-Score: {best_results['test_f1']:.4f}")
        print(f"Test ROC-AUC: {best_results['test_roc_auc']:.4f}")
        print(f"Cross-Validation ROC-AUC: {best_results['cv_mean']:.4f} (+/- {best_results['cv_std'] * 2:.4f})")
        
        if 'potential_savings' in self.results.get('business_impact', {}):
            print(f"Potential Annual Savings: ${self.results['business_impact']['potential_savings']:,.2f}")
        
        return self.results

if __name__ == "__main__":
    # Initialize churn prediction
    churn_predictor = AdvancedChurnPrediction('data/fintech_customer_data.csv')
    
    # Generate comprehensive report
    results = churn_predictor.generate_churn_prediction_report()
    
    print(f"\n[v0] Churn prediction model complete with {churn_predictor.best_model_name} achieving {results['model_comparison'][churn_predictor.best_model_name]['test_roc_auc']:.1%} ROC-AUC.")
    print(f"[v0] Model includes feature selection, hyperparameter optimization, and SHAP interpretability.")
    print(f"[v0] Business impact analysis shows potential savings from churn prevention initiatives.")
