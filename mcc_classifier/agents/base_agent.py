from abc import ABC, abstractmethod
import logging

class MCCClassifierAgent(ABC):
    """
    Abstract base class for MCC classification agents.
    
    All MCC classifier agents should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name):
        """
        Initialize the MCC classifier agent.
        
        Args:
            name (str): The name of the agent.
        """
        self.name = name
        self.logger = logging.getLogger(f"mcc_classifier.agent.{name}")
    
    @abstractmethod
    def classify(self, merchant_name, legal_name=None):
        """
        Classify a merchant based on its name and return the suggested MCC code.
        
        Args:
            merchant_name (str): The name of the merchant.
            legal_name (str, optional): The legal name of the merchant.
            
        Returns:
            dict: A dictionary containing:
                - 'mcc_code': The suggested MCC code.
                - 'mcc_description': The description of the MCC code.
                - 'confidence': A confidence score (0-1) for the classification.
                - 'alternative_mccs': Optional list of alternative MCCs with their confidence scores.
        """
        pass
    
    def get_performance_metrics(self, correct_classifications, total_classifications):
        """
        Calculate performance metrics for the agent.
        
        Args:
            correct_classifications (int): Number of correct classifications.
            total_classifications (int): Total number of classifications.
            
        Returns:
            dict: A dictionary containing performance metrics.
        """
        accuracy = correct_classifications / total_classifications if total_classifications > 0 else 0
        
        return {
            'agent_name': self.name,
            'accuracy': accuracy,
            'correct_classifications': correct_classifications,
            'total_classifications': total_classifications
        } 