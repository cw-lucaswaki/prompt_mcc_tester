import argparse
import logging
import os
import sys
from datetime import datetime

from mcc_classifier.evaluator import MCCEvaluator
from mcc_classifier.utils.logger import setup_logging

def parse_arguments():
    """
    Parse command-line arguments for the MCC classifier evaluation.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate MCC classification agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input CSV file containing merchant data"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to the output CSV file. If not provided, a default name will be used."
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to run the MCC classification evaluation.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level, log_to_file=not args.no_log_file)
    
    # Validate input file
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Set default output file if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        args.output = f"{output_dir}/mcc_evaluation_{timestamp}.csv"
    
    # Run evaluation
    try:
        logging.info("Starting MCC classification evaluation")
        
        evaluator = MCCEvaluator()
        metrics = evaluator.evaluate(args.input, args.output)
        
        # Print results summary
        logging.info("Evaluation completed successfully")
        logging.info(f"Results written to: {args.output}")
        
        for agent_name, agent_metrics in metrics.items():
            accuracy = agent_metrics["accuracy"]
            correct = agent_metrics["correct_classifications"]
            total = agent_metrics["total_classifications"]
            
            logging.info(f"{agent_name} - Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return 0
    
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 