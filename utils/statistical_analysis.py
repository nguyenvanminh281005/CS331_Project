"""
Statistical Analysis for Age-Invariant Face Recognition
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, List, Tuple
import config


class StatisticalAnalyzer:
    """Statistical analysis for face recognition results"""
    
    def __init__(self, confidence_level=0.95):
        """
        Args:
            confidence_level: Confidence level for intervals
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_confidence_interval(self, data: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate mean and confidence interval
        
        Args:
            data: Array of values
            
        Returns:
            mean, lower_bound, upper_bound
        """
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(self.confidence_level, len(data)-1, loc=mean, scale=sem)
        
        return mean, ci[0], ci[1]
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                     n_bootstrap=1000) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval
        
        Args:
            data: Array of values
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            mean, lower_bound, upper_bound
        """
        np.random.seed(config.SEED)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        mean = np.mean(data)
        lower = np.percentile(bootstrap_means, (self.alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - self.alpha/2) * 100)
        
        return mean, lower, upper
    
    def correlation_analysis(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Analyze correlation between two variables
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            correlation_results: Dictionary with correlation metrics
        """
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(x, y)
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(x, y)
        
        results = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'pearson_significant': pearson_p < 0.05,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'spearman_significant': spearman_p < 0.05,
        }
        
        return results
    
    def linear_mixed_effects_model(self, data_df: pd.DataFrame) -> Dict:
        """
        Fit Linear Mixed-Effects Model (LME) to analyze factors affecting match scores
        
        Model: match_score ~ time_gap + age + gender + (1 | identity)
        
        Args:
            data_df: DataFrame with columns [match_score, time_gap, age, gender, identity]
            
        Returns:
            model_results: Dictionary with model results
        """
        try:
            # Prepare data
            df = data_df.copy()
            
            # Ensure required columns exist
            required_cols = ['match_score', 'time_gap', 'identity']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found in dataframe")
                    return None
            
            # Fit mixed-effects model
            # Formula: match_score ~ time_gap + age + (1 | identity)
            if 'age' in df.columns and 'gender' in df.columns:
                formula = 'match_score ~ time_gap + age + gender'
            elif 'age' in df.columns:
                formula = 'match_score ~ time_gap + age'
            else:
                formula = 'match_score ~ time_gap'
            
            # Use MixedLM from statsmodels
            model = smf.mixedlm(formula, df, groups=df['identity'])
            result = model.fit()
            
            # Extract results
            results = {
                'summary': result.summary().as_text(),
                'coefficients': result.params.to_dict(),
                'p_values': result.pvalues.to_dict(),
                'confidence_intervals': result.conf_int().to_dict(),
                'aic': result.aic,
                'bic': result.bic,
            }
            
            # Identify significant factors
            significant_factors = []
            for factor, p_value in result.pvalues.items():
                if p_value < 0.05:
                    significant_factors.append(factor)
            
            results['significant_factors'] = significant_factors
            
            return results
            
        except Exception as e:
            print(f"Error fitting LME model: {e}")
            return None
    
    def analyze_degradation_by_demographic(self, data_df: pd.DataFrame) -> Dict:
        """
        Analyze performance degradation by demographic groups
        
        Args:
            data_df: DataFrame with columns [match_score, time_gap, age_group, gender, ...]
            
        Returns:
            analysis_results: Dictionary with degradation analysis by groups
        """
        results = {}
        
        # Analyze by age group
        if 'age_group' in data_df.columns:
            age_group_results = {}
            for age_group in data_df['age_group'].unique():
                group_data = data_df[data_df['age_group'] == age_group]
                
                # Calculate degradation rate
                if 'time_gap' in group_data.columns and 'match_score' in group_data.columns:
                    corr_result = self.correlation_analysis(
                        group_data['time_gap'].values,
                        group_data['match_score'].values
                    )
                    age_group_results[age_group] = corr_result
            
            results['by_age_group'] = age_group_results
        
        # Analyze by gender
        if 'gender' in data_df.columns:
            gender_results = {}
            for gender in data_df['gender'].unique():
                group_data = data_df[data_df['gender'] == gender]
                
                if 'time_gap' in group_data.columns and 'match_score' in group_data.columns:
                    corr_result = self.correlation_analysis(
                        group_data['time_gap'].values,
                        group_data['match_score'].values
                    )
                    gender_results[gender] = corr_result
            
            results['by_gender'] = gender_results
        
        return results
    
    def hypothesis_test_degradation(self, baseline_scores: np.ndarray,
                                   temporal_scores: np.ndarray) -> Dict:
        """
        Hypothesis test: Does temporal-aware model reduce degradation?
        
        H0: No difference between baseline and temporal model
        H1: Temporal model has better performance
        
        Args:
            baseline_scores: Scores from baseline model
            temporal_scores: Scores from temporal-aware model
            
        Returns:
            test_results: Dictionary with test results
        """
        # Paired t-test (if same pairs)
        if len(baseline_scores) == len(temporal_scores):
            t_stat, p_value = stats.ttest_rel(temporal_scores, baseline_scores)
            test_type = 'paired_t_test'
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(temporal_scores, baseline_scores)
            test_type = 'independent_t_test'
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        if len(baseline_scores) == len(temporal_scores):
            w_stat, w_p_value = stats.wilcoxon(temporal_scores, baseline_scores)
        else:
            # Mann-Whitney U test
            w_stat, w_p_value = stats.mannwhitneyu(temporal_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(temporal_scores)**2) / 2)
        cohens_d = (np.mean(temporal_scores) - np.mean(baseline_scores)) / pooled_std
        
        results = {
            'test_type': test_type,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_p_value': w_p_value,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large'),
            'baseline_mean': np.mean(baseline_scores),
            'baseline_std': np.std(baseline_scores),
            'temporal_mean': np.mean(temporal_scores),
            'temporal_std': np.std(temporal_scores),
        }
        
        return results
    
    def regression_analysis(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Linear regression analysis
        
        Args:
            x: Independent variable (e.g., time gap)
            y: Dependent variable (e.g., TAR)
            
        Returns:
            regression_results: Dictionary with regression results
        """
        # Add constant for intercept
        X = sm.add_constant(x)
        
        # Fit model
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Extract results
        regression_results = {
            'slope': results.params[1],
            'intercept': results.params[0],
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'p_value': results.pvalues[1],
            'confidence_interval': results.conf_int().iloc[1].tolist(),
            'summary': results.summary().as_text(),
        }
        
        return regression_results
    
    def anova_analysis(self, groups: List[np.ndarray]) -> Dict:
        """
        One-way ANOVA to compare multiple groups
        
        Args:
            groups: List of arrays (one per group)
            
        Returns:
            anova_results: Dictionary with ANOVA results
        """
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Post-hoc Tukey HSD test (if significant)
        results = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'num_groups': len(groups),
        }
        
        # Group statistics
        group_stats = []
        for i, group in enumerate(groups):
            group_stats.append({
                'group': i,
                'mean': np.mean(group),
                'std': np.std(group),
                'n': len(group),
            })
        results['group_statistics'] = group_stats
        
        return results


def create_statistical_report(results_dict: Dict, output_path: str):
    """
    Create comprehensive statistical analysis report
    
    Args:
        results_dict: Dictionary with all statistical results
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("Age-Invariant Face Recognition\n")
        f.write("="*80 + "\n\n")
        
        for analysis_name, results in results_dict.items():
            f.write(f"\n{analysis_name}\n")
            f.write("-"*80 + "\n")
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
                    elif isinstance(value, str):
                        f.write(f"  {key}: {value}\n")
                    elif isinstance(value, list):
                        f.write(f"  {key}: {value}\n")
            
            f.write("\n")
    
    print(f"Statistical report saved to {output_path}")


if __name__ == "__main__":
    # Test statistical analysis
    print("Testing statistical analysis...")
    
    np.random.seed(42)
    
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    
    # Test confidence interval
    data = np.random.normal(0.85, 0.05, 100)
    mean, lower, upper = analyzer.calculate_confidence_interval(data)
    print(f"\nConfidence Interval:")
    print(f"  Mean: {mean:.4f}")
    print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
    
    # Test correlation
    time_gaps = np.array([1, 2, 4, 6, 8, 10])
    tar_values = np.array([0.95, 0.92, 0.88, 0.84, 0.80, 0.76])
    
    corr_results = analyzer.correlation_analysis(time_gaps, tar_values)
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson r: {corr_results['pearson_r']:.4f} (p={corr_results['pearson_p']:.4f})")
    print(f"  Spearman r: {corr_results['spearman_r']:.4f} (p={corr_results['spearman_p']:.4f})")
    
    # Test hypothesis testing
    baseline_scores = np.random.normal(0.85, 0.05, 100)
    temporal_scores = np.random.normal(0.88, 0.05, 100)
    
    hyp_results = analyzer.hypothesis_test_degradation(baseline_scores, temporal_scores)
    print(f"\nHypothesis Test:")
    print(f"  T-statistic: {hyp_results['t_statistic']:.4f}")
    print(f"  P-value: {hyp_results['p_value']:.4f}")
    print(f"  Significant: {hyp_results['significant']}")
    print(f"  Cohen's d: {hyp_results['cohens_d']:.4f} ({hyp_results['effect_size']})")
    
    # Test regression
    reg_results = analyzer.regression_analysis(time_gaps, tar_values)
    print(f"\nRegression Analysis:")
    print(f"  Slope: {reg_results['slope']:.4f}")
    print(f"  Intercept: {reg_results['intercept']:.4f}")
    print(f"  R²: {reg_results['r_squared']:.4f}")
    print(f"  P-value: {reg_results['p_value']:.4f}")
    
    print("\nStatistical analysis tests complete!")
