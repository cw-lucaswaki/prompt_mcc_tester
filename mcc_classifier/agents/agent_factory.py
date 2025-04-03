import logging
from mcc_classifier.agents.rafa_agent import RafaAgent
from mcc_classifier.agents.matheus_agent import MatheusAgent
from mcc_classifier.agents.waki_agent import WakiAgent

logger = logging.getLogger("mcc_classifier.agent_factory")

class AgentFactory:
    """
    Factory class for creating MCC classifier agents.
    
    This class provides a central way to instantiate different types of
    MCC classifier agents while hiding implementation details.
    """
    
    @staticmethod
    def create_agent(agent_type):
        """
        Create an instance of the specified agent type.
        
        Args:
            agent_type (str): The type of agent to create ("rafa", "matheus", or "waki").
            
        Returns:
            MCCClassifierAgent: An instance of the specified agent type.
            
        Raises:
            ValueError: If an unsupported agent type is specified.
        """
        agent_type = agent_type.lower()
        
        if agent_type == "rafa":
            logger.info("Creating Rafa agent")
            return RafaAgent()
        elif agent_type == "matheus":
            logger.info("Creating Matheus agent")
            return MatheusAgent()
        elif agent_type == "waki":
            logger.info("Creating Waki agent")
            return WakiAgent()
        else:
            error_msg = f"Unsupported agent type: {agent_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def create_all_agents():
        """
        Create instances of all available agents.
        
        Returns:
            list: A list containing instances of all available agents.
        """
        logger.info("Creating all agents")
        
        return [
            RafaAgent(),
            MatheusAgent(),
            WakiAgent()
        ] 