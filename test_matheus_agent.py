#!/usr/bin/env python3
"""
Test MatheusAgent MCC Classification

This script specifically tests the MatheusAgent's performance in classifying
merchants with their correct Merchant Category Codes (MCCs).
"""

import os
import csv
import logging
from datetime import datetime
from mcc_classifier.agents.matheus_agent import MatheusAgent
from mcc_classifier.utils.logger import setup_logging

def test_matheus_agent(input_file):
    """
    Test MatheusAgent on the specified input file.
    
    Args:
        input_file (str): Path to input CSV file
    """
    # Configure logging
    setup_logging(log_level=logging.INFO, log_to_file=True)
    logger = logging.getLogger("test_matheus_agent")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    # Set output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/matheus_agent_test_{timestamp}.csv"
    
    # Initialize the agent
    agent = MatheusAgent()
    
    # Read input data
    merchants = []
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                merchants.append(row)
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
    
    # Process merchants
    results = []
    correct_count = 0
    total_count = 0
    
    for merchant in merchants:
        merchant_name = merchant.get("Merchant Name", "")
        legal_name = merchant.get("Legal Name", "")
        actual_mcc = merchant.get("Actual MCC code", "")
        mcc_description = merchant.get("MCC Description", "")
        
        if not merchant_name or not actual_mcc:
            print(f"Skipping row with missing data: {merchant}")
            continue
        
        print(f"\nProcessing: {merchant_name}")
        
        # Classify merchant
        try:
            result = agent.classify(merchant_name, legal_name)
            
            # Check if correct
            is_correct = str(result["mcc_code"]).strip() == str(actual_mcc).strip()
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Store result
            results.append({
                "Merchant Name": merchant_name,
                "Legal Name": legal_name,
                "Actual MCC": actual_mcc, 
                "Actual MCC Description": mcc_description,
                "Suggested MCC": result["mcc_code"],
                "Suggested MCC Description": result["mcc_description"],
                "Confidence": result["confidence"],
                "Is Correct": "Yes" if is_correct else "No"
            })
            
            # Print result
            print(f"  Actual MCC: {actual_mcc} ({mcc_description})")
            print(f"  Suggested MCC: {result['mcc_code']} ({result['mcc_description']})")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Correct: {'Yes' if is_correct else 'No'}")
            
            # Print analysis if available
            if "analysis" in result:
                print("  Analysis:")
                for point in result["analysis"]:
                    print(f"    - {point}")
                    
        except Exception as e:
            logger.error(f"Error classifying merchant {merchant_name}: {str(e)}")
            print(f"  Error: {str(e)}")
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # Write results to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                print(f"Results written to: {output_file}")
            else:
                print("No results to write to file.")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")
    
if __name__ == "__main__":
    # Run test on custom test data
    test_matheus_agent("data/custom_test_merchants.csv") 