import pandas as pd
import pytest
from graph_optim.optimizer import DataFrameOptimizer

def test_optimizer_convergence():
    df = pd.DataFrame({
        'value': [10, 15, 20],
        'target': [12, 18, 22]
    })
    
    optimizer = DataFrameOptimizer(max_iterations=10, convergence_threshold=0.1)
    result_df = optimizer.optimize(df)
    
    # Check that values are closer to targets
    initial_diff = abs(df['value'] - df['target']).mean()
    final_diff = abs(result_df['value'] - result_df['target']).mean()
    
    assert final_diff < initial_diff

def test_optimizer_max_iterations():
    df = pd.DataFrame({
        'value': [10, 15, 20],
        'target': [12, 18, 22]
    })
    
    max_iterations = 3
    optimizer = DataFrameOptimizer(max_iterations=max_iterations, convergence_threshold=0.01)
    
    # Run optimization
    result = optimizer.workflow.invoke({
        "df": df,
        "iteration": 0,
        "converged": False
    })
    
    assert result["iteration"] <= max_iterations 