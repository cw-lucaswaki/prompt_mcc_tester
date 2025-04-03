import os
import argparse
import logging
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from mcc_classifier.agents.agent_factory import AgentFactory
from mcc_classifier.evaluator import MCCEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("run_waki_comparison")

def create_test_data(output_file):
    """
    Create a test data CSV file with merchant examples.
    
    Args:
        output_file (str): Path to save the test data CSV
    """
    test_data = [
        {"Merchant Name": "City Grocery", "Legal Name": "City Grocery Inc.", "Actual MCC code": "5411", "MCC Description": "Grocery Stores, Supermarkets"},
        {"Merchant Name": "Elite Electronics", "Legal Name": "Elite Electronics LLC", "Actual MCC code": "5732", "MCC Description": "Electronics Sales"},
        {"Merchant Name": "Joe's Pizza", "Legal Name": "Joe's Pizza Co.", "Actual MCC code": "5812", "MCC Description": "Eating places and Restaurants"},
        {"Merchant Name": "ABC Hardware", "Legal Name": "ABC Hardware Supply", "Actual MCC code": "5251", "MCC Description": "Hardware Stores"},
        {"Merchant Name": "Sunny Day Cafe", "Legal Name": "Sunny Day Enterprises", "Actual MCC code": "5812", "MCC Description": "Eating places and Restaurants"},
        {"Merchant Name": "Quick Mart", "Legal Name": "Quick Mart Inc.", "Actual MCC code": "5499", "MCC Description": "Misc. Food Stores – Convenience Stores and Specialty Markets"},
        {"Merchant Name": "The Bike Shop", "Legal Name": "Cycle City LLC", "Actual MCC code": "5940", "MCC Description": "Bicycle Shops – Sales and Service"},
        {"Merchant Name": "Maria's Salon", "Legal Name": "Maria's Beauty LLC", "Actual MCC code": "7230", "MCC Description": "Barber and Beauty Shops"},
        {"Merchant Name": "Downtown Dental Clinic", "Legal Name": "Downtown Dental Group", "Actual MCC code": "8021", "MCC Description": "Dentists and Orthodontists"},
        {"Merchant Name": "Pet Paradise", "Legal Name": "Pet Supplies Co.", "Actual MCC code": "5995", "MCC Description": "Pet Shops, Pet Foods, and Supplies Stores"},
        {"Merchant Name": "The Fitness Center", "Legal Name": "Health & Wellness LLC", "Actual MCC code": "7997", "MCC Description": "Membership Clubs (Sports, Recreation, Athletic), Country Clubs, and Private Golf Courses"},
        {"Merchant Name": "John Smith", "Legal Name": "John Smith Consulting LLC", "Actual MCC code": "7399", "MCC Description": "Business Services, Not Elsewhere Classified"},
        {"Merchant Name": "Tech Solutions", "Legal Name": "Digital Services Inc.", "Actual MCC code": "7372", "MCC Description": "Computer Programming, Integrated Systems Design and Data Processing Services"},
        {"Merchant Name": "Sarah's Bakery", "Legal Name": "Sweet Treats Inc.", "Actual MCC code": "5462", "MCC Description": "Bakeries"},
        {"Merchant Name": "Green Thumb Landscaping", "Legal Name": "Green Thumb Services", "Actual MCC code": "0780", "MCC Description": "Horticultural Services, Landscaping Services"}
    ]
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(test_data)
    df.to_csv(output_file, index=False)
    logger.info(f"Created test data file: {output_file}")
    return output_file

def run_comparison(input_file=None, output_dir="output"):
    """
    Run a comparison between all MCC classification agents.
    
    Args:
        input_file (str, optional): Path to input CSV file. If not provided, a test file will be created.
        output_dir (str): Directory to save output files
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create test data if input file not provided
    if not input_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = os.path.join(output_dir, f"test_merchants_{timestamp}.csv")
        create_test_data(input_file)
    
    # Generate output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"comparison_results_{timestamp}.csv")
    
    # Create an instance of the evaluator with all agents
    evaluator = MCCEvaluator()
    
    # Run the evaluation
    logger.info(f"Running comparison with input file: {input_file}")
    metrics = evaluator.evaluate(input_file, output_file)
    
    # Display results
    print("\n===== Comparison Results =====")
    for agent_name, agent_metrics in metrics.items():
        print(f"\nAgent: {agent_name}")
        print(f"Accuracy: {agent_metrics['accuracy']:.2%}")
        print(f"Correct: {agent_metrics['correct_classifications']}/{agent_metrics['total_classifications']}")
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save metrics to JSON for further analysis
    metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Run comparison between MCC classification agents")
    parser.add_argument("--input", type=str, help="Path to input CSV file with merchant data")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    args = parser.parse_args()
    
    # Ensure OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set. GPT-based agents will use fallback classification.")
    
    # Run the comparison
    output_file = run_comparison(args.input, args.output_dir)
    print(f"\nComparison complete. Results saved to: {output_file}")

if __name__ == "__main__":
    main() 