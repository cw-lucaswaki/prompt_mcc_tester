import logging
import os
from typing import List, Dict, Any

from mcc_classifier.agents.agent_factory import AgentFactory
from mcc_classifier.agents.base_agent import MCCClassifierAgent
from mcc_classifier.utils.data_handler import DataHandler

logger = logging.getLogger("mcc_classifier.evaluator")

class MCCEvaluator:
    """
    Main class for evaluating MCC classification agents.
    
    This class orchestrates the process of evaluating different MCC classification
    agents against a dataset of merchants with known MCCs.
    """
    
    def __init__(self, agents: List[MCCClassifierAgent] = None):
        """
        Initialize the MCC evaluator.
        
        Args:
            agents (list, optional): A list of MCC classifier agents to evaluate.
                If not provided, all available agents will be used.
        """
        self.agents = agents or AgentFactory.create_all_agents()
        logger.info(f"Initialized MCC evaluator with {len(self.agents)} agents")
    
    def evaluate(self, input_file: str, output_file: str, pass_full_data: bool = False) -> Dict[str, Any]:
        """
        Evaluate the performance of MCC classification agents on a dataset.
        
        Args:
            input_file (str): The path to the input CSV file containing merchant data.
            output_file (str): The path to write the output CSV file with evaluation results.
            pass_full_data (bool): Whether to pass full merchant data to agents.
            
        Returns:
            dict: A dictionary containing evaluation metrics for each agent.
            
        Raises:
            FileNotFoundError: If the input file does not exist.
            Exception: For other errors encountered during evaluation.
        """
        try:
            logger.info(f"Starting evaluation with input file: {input_file}")
            
            # Read input data
            input_data = DataHandler.read_csv(input_file)
            logger.info(f"Read {len(input_data)} merchants from input file")
            
            # Prepare output data
            output_data = []
            
            # Set up metrics tracking
            metrics = {agent.name: {"correct": 0, "total": 0} for agent in self.agents}
            
            # Process each merchant
            for merchant in input_data:
                merchant_name = merchant.get("Merchant Name", "")
                legal_name = merchant.get("Legal Name", "")
                actual_mcc = merchant.get("Actual MCC code", "")
                mcc_description = merchant.get("MCC Description", "")
                original_mcc_code = merchant.get("original Mcc code", "")
                ai_original_description = merchant.get("ai_original_description", "")
                
                # Skip rows with missing data
                if not merchant_name or not actual_mcc:
                    logger.warning(f"Skipping row with missing data: {merchant}")
                    continue
                
                # Create output row with base data
                output_row = {
                    "Merchant Name": merchant_name,
                    "Legal Name": legal_name,
                    "Actual MCC": actual_mcc,
                    "MCC Description": mcc_description
                }
                
                # Prepare additional data to pass to the agent
                additional_data = {
                    "original_mcc_code": original_mcc_code,
                    "mcc_description": mcc_description,
                    "ai_original_description": ai_original_description,
                    # Include any other fields that might be useful
                    **{k: v for k, v in merchant.items() if k not in ["Merchant Name", "Legal Name"]}
                }
                
                # Get classifications from each agent
                for agent in self.agents:
                    try:
                        # Try to classify merchant with appropriate method based on pass_full_data flag
                        if pass_full_data:
                            try:
                                # First try with full data
                                result = agent.classify(merchant_name, legal_name, **additional_data)
                            except TypeError as e:
                                # If the agent doesn't support full data, fall back to basic parameters
                                logger.info(f"Agent {agent.name} doesn't support full data, falling back to basic parameters")
                                result = agent.classify(merchant_name, legal_name)
                        else:
                            # Use the simple classify method with just merchant name and legal name
                            result = agent.classify(merchant_name, legal_name)
                        
                        # Add result to output row
                        output_row[f"{agent.name}'s suggested MCC"] = result["mcc_code"]
                        output_row[f"{agent.name}'s MCC description"] = result["mcc_description"]
                        output_row[f"{agent.name}'s confidence"] = result["confidence"]
                        
                        # Update metrics
                        metrics[agent.name]["total"] += 1
                        if str(result["mcc_code"]).strip() == str(actual_mcc).strip():
                            metrics[agent.name]["correct"] += 1
                            output_row[f"{agent.name}'s match"] = "Yes"
                        else:
                            output_row[f"{agent.name}'s match"] = "No"
                    
                    except Exception as e:
                        logger.error(f"Error classifying merchant {merchant_name} with agent {agent.name}: {str(e)}")
                        output_row[f"{agent.name}'s suggested MCC"] = "ERROR"
                        output_row[f"{agent.name}'s MCC description"] = str(e)
                        output_row[f"{agent.name}'s confidence"] = 0.0
                        output_row[f"{agent.name}'s match"] = "Error"
                
                # Add row to output data
                output_data.append(output_row)
            
            # Calculate performance metrics
            performance_metrics = {}
            for agent_name, data in metrics.items():
                correct = data["correct"]
                total = data["total"]
                performance_metrics[agent_name] = {
                    "accuracy": correct / total if total > 0 else 0,
                    "correct_classifications": correct,
                    "total_classifications": total
                }
            
            # Add summary row
            summary_row = {
                "Merchant Name": "SUMMARY",
                "Legal Name": "",
                "Actual MCC": "",
                "MCC Description": ""
            }
            
            for agent in self.agents:
                agent_metrics = performance_metrics[agent.name]
                summary_row[f"{agent.name}'s suggested MCC"] = ""
                summary_row[f"{agent.name}'s MCC description"] = ""
                summary_row[f"{agent.name}'s confidence"] = ""
                summary_row[f"{agent.name}'s match"] = f"Accuracy: {agent_metrics['accuracy']:.2%}"
            
            output_data.append(summary_row)
            
            # Write output data
            DataHandler.write_csv(output_file, output_data)
            
            logger.info(f"Evaluation complete. Results written to {output_file}")
            
            # Return performance metrics
            return performance_metrics
        
        except Exception as e:
            error_msg = f"Error during evaluation: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) 