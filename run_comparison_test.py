#!/usr/bin/env python3
"""
MCC Classification Comparison Tool

This script runs the MCC classification on multiple datasets to compare results
across different test scenarios.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from mcc_classifier.evaluator import MCCEvaluator
from mcc_classifier.utils.logger import setup_logging

def run_comparison(datasets, sample_only=False):
    """
    Run MCC evaluation on multiple datasets and compare results.
    
    Args:
        datasets (list): List of dataset filenames to process
        sample_only (bool): If True, only run on sample datasets
    """
    # Configure logging
    setup_logging(log_level=logging.INFO, log_to_file=True)
    
    for dataset in datasets:
        # Skip non-sample datasets if sample_only is True
        if sample_only and not dataset.startswith("sample_"):
            continue
            
        input_file = os.path.join("data", dataset)
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            continue
        
        # Set output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = f"{output_dir}/{os.path.splitext(dataset)[0]}_comparison_{timestamp}.csv"
        
        # Run evaluation
        try:
            print(f"\n=== Running evaluation on {dataset} ===")
            
            evaluator = MCCEvaluator()
            # Pass pass_full_data=True to ensure the agent receives all merchant data
            metrics = evaluator.evaluate(input_file, output_file, pass_full_data=True)
            
            # Print results summary
            print(f"Evaluation completed for {dataset}")
            print(f"Results written to: {output_file}")
            
            # Print performance table header
            print("\n| Agent    | Accuracy | Correct/Total |")
            print("|----------|----------|---------------|")
            
            # Print each agent's performance
            for agent_name, agent_metrics in metrics.items():
                accuracy = agent_metrics["accuracy"]
                correct = agent_metrics["correct_classifications"]
                total = agent_metrics["total_classifications"]
                
                print(f"| {agent_name:<8} | {accuracy:.2%}   | {correct}/{total}           |")
            
        except Exception as e:
            print(f"Error during evaluation of {dataset}: {str(e)}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MCC Classification Comparison Tool')
    parser.add_argument('--sample', action='store_true', 
                        help='Run only on sample datasets')
    args = parser.parse_args()
    
    # Datasets to compare
    datasets_to_compare = [
        # "sample_merchants.csv",
        "custom_test_merchants.csv"
    ]
    
    # Run comparison with the sample flag
    run_comparison(datasets_to_compare, sample_only=args.sample) 