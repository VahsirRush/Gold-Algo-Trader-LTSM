"""
MLflow experiment tracking for the gold trading framework.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import pandas as pd
import numpy as np

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class MLflowTracker:
    """MLflow experiment tracking for trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLflow tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not available. Install with: pip install mlflow")
            return
        
        # MLflow configuration
        self.tracking_uri = config.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.experiment_name = config.get('MLFLOW_EXPERIMENT_NAME', 'gold-trading')
        self.artifact_location = config.get('MLFLOW_ARTIFACT_LOCATION', './mlruns')
        
        # Initialize MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=self.artifact_location
                )
            
            mlflow.set_experiment(self.experiment_name)
            self.logger.info(f"MLflow tracking initialized: {self.tracking_uri}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup MLflow: {e}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
            
        Returns:
            Run ID
        """
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            # Start run
            mlflow.start_run(run_name=run_name)
            
            # Add tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Add default tags
            mlflow.set_tag("framework", "gold-algo")
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            run_id = mlflow.active_run().info.run_id
            self.logger.info(f"Started MLflow run: {run_id}")
            return run_id
            
        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {e}")
            return None
    
    def end_run(self):
        """End the current MLflow run."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.end_run()
            self.logger.info("Ended MLflow run")
        except Exception as e:
            self.logger.error(f"Failed to end MLflow run: {e}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Parameters to log
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.log_params(params)
            self.logger.debug(f"Logged parameters: {list(params.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Metrics to log
            step: Step number (optional)
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
            self.logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def log_artifacts(self, local_path: str, artifact_path: str = None):
        """
        Log artifacts to MLflow.
        
        Args:
            local_path: Local path to artifact
            artifact_path: Path within the run's artifact directory
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            self.logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            self.logger.error(f"Failed to log artifact {local_path}: {e}")
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn"):
        """
        Log a model to MLflow.
        
        Args:
            model: Model object to log
            model_name: Name for the model
            model_type: Type of model ('sklearn', 'pytorch', etc.)
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, model_name)
            else:
                mlflow.log_model(model, model_name)
            
            self.logger.info(f"Logged {model_type} model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to log model {model_name}: {e}")
    
    def log_backtest_results(self, results: Dict[str, Any], strategy_name: str):
        """
        Log backtest results to MLflow.
        
        Args:
            results: Backtest results dictionary
            strategy_name: Name of the strategy
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            # Log metrics
            metrics = {
                'total_return': results.get('total_return', 0),
                'annual_return': results.get('annual_return', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'volatility': results.get('volatility', 0),
                'num_trades': results.get('num_trades', 0),
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0)
            }
            
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            
            self.log_metrics(metrics)
            
            # Log parameters
            params = {
                'strategy': strategy_name,
                'initial_capital': results.get('initial_capital', 0),
                'commission': results.get('commission', 0),
                'slippage': results.get('slippage', 0)
            }
            
            self.log_parameters(params)
            
            # Log equity curve if available
            if 'equity_curve' in results and results['equity_curve']:
                equity_df = pd.DataFrame({
                    'equity': results['equity_curve']
                })
                equity_path = f"equity_curve_{strategy_name}.csv"
                equity_df.to_csv(equity_path, index=False)
                self.log_artifacts(equity_path, "equity_curves")
                os.remove(equity_path)
            
            # Log daily returns if available
            if 'daily_returns' in results and results['daily_returns']:
                returns_df = pd.DataFrame({
                    'returns': results['daily_returns']
                })
                returns_path = f"daily_returns_{strategy_name}.csv"
                returns_df.to_csv(returns_path, index=False)
                self.log_artifacts(returns_path, "daily_returns")
                os.remove(returns_path)
            
            self.logger.info(f"Logged backtest results for {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to log backtest results: {e}")
    
    def log_optimization_results(self, results: List[Dict[str, Any]], strategy_name: str):
        """
        Log optimization results to MLflow.
        
        Args:
            results: List of optimization results
            strategy_name: Name of the strategy
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            if not results:
                return
            
            # Log best result
            best_result = results[0]
            
            # Log best parameters
            best_params = best_result.get('parameters', {})
            self.log_parameters(best_params)
            
            # Log best metrics
            best_metrics = {
                'best_total_return': best_result.get('total_return', 0),
                'best_sharpe_ratio': best_result.get('sharpe_ratio', 0),
                'best_max_drawdown': best_result.get('max_drawdown', 0),
                'best_win_rate': best_result.get('win_rate', 0),
                'best_profit_factor': best_result.get('profit_factor', 0)
            }
            
            best_metrics = {k: v for k, v in best_metrics.items() if v is not None}
            self.log_metrics(best_metrics)
            
            # Log optimization statistics
            all_returns = [r.get('total_return', 0) for r in results if r.get('total_return') is not None]
            all_sharpes = [r.get('sharpe_ratio', 0) for r in results if r.get('sharpe_ratio') is not None]
            
            if all_returns:
                opt_stats = {
                    'mean_return': np.mean(all_returns),
                    'std_return': np.std(all_returns),
                    'min_return': np.min(all_returns),
                    'max_return': np.max(all_returns),
                    'mean_sharpe': np.mean(all_sharpes),
                    'std_sharpe': np.std(all_sharpes),
                    'num_combinations': len(results)
                }
                self.log_metrics(opt_stats)
            
            # Save all results as artifact
            results_path = f"optimization_results_{strategy_name}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.log_artifacts(results_path, "optimization_results")
            os.remove(results_path)
            
            self.logger.info(f"Logged optimization results for {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to log optimization results: {e}")
    
    def log_live_trading_results(self, results: Dict[str, Any]):
        """
        Log live trading results to MLflow.
        
        Args:
            results: Live trading results dictionary
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            # Log metrics
            metrics = {
                'live_total_return': results.get('total_return', 0),
                'live_sharpe_ratio': results.get('sharpe_ratio', 0),
                'live_max_drawdown': results.get('max_drawdown', 0),
                'live_num_trades': results.get('num_trades', 0),
                'live_win_rate': results.get('win_rate', 0),
                'live_profit_factor': results.get('profit_factor', 0)
            }
            
            metrics = {k: v for k, v in metrics.items() if v is not None}
            self.log_metrics(metrics)
            
            # Log parameters
            params = {
                'trading_mode': 'live',
                'broker': results.get('broker', 'unknown'),
                'symbol': results.get('symbol', 'unknown')
            }
            
            self.log_parameters(params)
            
            # Log trading log if available
            if 'trading_log' in results:
                log_path = "trading_log.csv"
                results['trading_log'].to_csv(log_path, index=False)
                self.log_artifacts(log_path, "trading_logs")
                os.remove(log_path)
            
            self.logger.info("Logged live trading results")
            
        except Exception as e:
            self.logger.error(f"Failed to log live trading results: {e}")
    
    def log_risk_metrics(self, risk_metrics: Dict[str, float]):
        """
        Log risk metrics to MLflow.
        
        Args:
            risk_metrics: Risk metrics dictionary
        """
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            # Add prefix to risk metrics
            prefixed_metrics = {f"risk_{k}": v for k, v in risk_metrics.items() if v is not None}
            self.log_metrics(prefixed_metrics)
            
            self.logger.debug(f"Logged risk metrics: {list(risk_metrics.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to log risk metrics: {e}")
    
    def get_experiment_runs(self, experiment_name: str = None) -> List[Dict[str, Any]]:
        """
        Get runs from an experiment.
        
        Args:
            experiment_name: Name of the experiment (optional)
            
        Returns:
            List of run information
        """
        if not MLFLOW_AVAILABLE:
            return []
        
        try:
            exp_name = experiment_name or self.experiment_name
            experiment = mlflow.get_experiment_by_name(exp_name)
            
            if experiment is None:
                return []
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                output_format="list"
            )
            
            run_info = []
            for run in runs:
                run_info.append({
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', ''),
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'metrics': run.data.metrics,
                    'params': run.data.params
                })
            
            return run_info
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {e}")
            return []
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison results
        """
        if not MLFLOW_AVAILABLE:
            return {}
        
        try:
            comparison = mlflow.compare_runs(run_ids)
            
            return {
                'run_ids': run_ids,
                'metrics_comparison': comparison.to_dict() if hasattr(comparison, 'to_dict') else str(comparison)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare runs: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.end_run() 