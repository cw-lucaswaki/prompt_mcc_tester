#!/usr/bin/env python3
"""
Custom MCC Classification Evaluation Tool

This script evaluates the performance of different AI agents in classifying
merchants with their correct Merchant Category Codes (MCCs).
"""

import os
import logging
from datetime import datetime
from mcc_classifier.evaluator import MCCEvaluator
from mcc_classifier.utils.logger import setup_logging

def run_evaluation(input_file):
    """
    Run MCC evaluation on the specified input file.
    
    Args:
        input_file (str): Path to input CSV file
    """
    # Configure logging
    setup_logging(log_level=logging.INFO, log_to_file=True)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    # Set output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/custom_mcc_evaluation_{timestamp}.csv"
    
    # Run evaluation
    try:
        print("Starting MCC classification evaluation")
        
        evaluator = MCCEvaluator()
        metrics = evaluator.evaluate(input_file, output_file)
        
        # Print results summary
        print("Evaluation completed successfully")
        print(f"Results written to: {output_file}")
        
        for agent_name, agent_metrics in metrics.items():
            accuracy = agent_metrics["accuracy"]
            correct = agent_metrics["correct_classifications"]
            total = agent_metrics["total_classifications"]
            
            print(f"{agent_name} - Accuracy: {accuracy:.2%} ({correct}/{total})")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    # Run evaluation on custom test data
    run_evaluation("data/custom_test_merchants.csv") 