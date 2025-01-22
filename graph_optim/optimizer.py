import pandas as pd
from typing import Dict, TypedDict
from langgraph.graph import Graph
import numpy as np

class OptimizationState(TypedDict):
    df: pd.DataFrame
    iteration: int
    converged: bool

class DataFrameOptimizer:
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.1):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.workflow = self._build_workflow()

    def check_convergence(self, state: OptimizationState) -> str:
        """Check if optimization has converged"""
        if state["iteration"] >= self.max_iterations or state["converged"]:
            return "end"
        return "continue"

    def optimize_step(self, state: OptimizationState) -> OptimizationState:
        """Perform one optimization step"""
        df = state["df"].copy()
        
        # Simple optimization: move values closer to targets
        df['value'] = df['value'] + 0.1 * (df['target'] - df['value'])
        
        # Check if we've converged
        mean_diff = np.abs(df['value'] - df['target']).mean()
        converged = mean_diff < self.convergence_threshold
        
        return {
            "df": df,
            "iteration": state["iteration"] + 1,
            "converged": converged
        }

    def _build_workflow(self) -> Graph:
        """Build the optimization workflow"""
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("optimize", self.optimize_step)
        
        # Add edges
        workflow.add_edge("optimize", "optimize")
        
        # Set the entry point
        workflow.set_entry_point("optimize")
        
        # Set the conditional edges
        workflow.add_conditional_edges(
            "optimize",
            self.check_convergence,
            {
                "continue": "optimize",
                "end": None
            }
        )
        
        return workflow.compile()

    def optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run optimization on the input DataFrame"""
        initial_state = {
            "df": df,
            "iteration": 0,
            "converged": False
        }
        
        result = self.workflow.invoke(initial_state)
        return result["df"] 