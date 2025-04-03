import pytest

from mcc_classifier.agents.agent_factory import AgentFactory
from mcc_classifier.agents.base_agent import MCCClassifierAgent
from mcc_classifier.agents.rafa_agent import RafaAgent
from mcc_classifier.agents.matheus_agent import MatheusAgent
from mcc_classifier.agents.waki_agent import WakiAgent

def test_agent_factory_create_all():
    """Test that the agent factory creates all agents."""
    agents = AgentFactory.create_all_agents()
    assert len(agents) == 3, "Should create 3 agents"
    
    agent_names = [agent.name for agent in agents]
    assert "Rafa" in agent_names
    assert "Matheus" in agent_names
    assert "Waki" in agent_names

def test_agent_factory_create_specific():
    """Test that the agent factory creates specific agents."""
    rafa = AgentFactory.create_agent("rafa")
    assert isinstance(rafa, RafaAgent)
    assert rafa.name == "Rafa"
    
    matheus = AgentFactory.create_agent("matheus")
    assert isinstance(matheus, MatheusAgent)
    assert matheus.name == "Matheus"
    
    waki = AgentFactory.create_agent("waki")
    assert isinstance(waki, WakiAgent)
    assert waki.name == "Waki"

def test_agent_factory_invalid_agent():
    """Test that the agent factory raises an error for invalid agent types."""
    with pytest.raises(ValueError):
        AgentFactory.create_agent("invalid")

def test_agent_classification_format():
    """Test that all agents return the correct classification format."""
    agents = AgentFactory.create_all_agents()
    
    for agent in agents:
        result = agent.classify("Test Merchant")
        
        # Check that the result has the required keys
        assert "mcc_code" in result
        assert "mcc_description" in result
        assert "confidence" in result
        assert "alternative_mccs" in result
        
        # Check that the confidence is a float between 0 and 1
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
        
        # Check that alternative_mccs is a list
        assert isinstance(result["alternative_mccs"], list)
        
        # Check each alternative MCC
        for alt_mcc in result["alternative_mccs"]:
            assert "mcc_code" in alt_mcc
            assert "mcc_description" in alt_mcc
            assert "confidence" in alt_mcc
            assert isinstance(alt_mcc["confidence"], float)
            assert 0 <= alt_mcc["confidence"] <= 1

def test_performance_metrics():
    """Test the calculation of performance metrics."""
    agent = RafaAgent()
    
    # Test with no classifications
    metrics = agent.get_performance_metrics(0, 0)
    assert metrics["accuracy"] == 0
    
    # Test with some correct classifications
    metrics = agent.get_performance_metrics(5, 10)
    assert metrics["accuracy"] == 0.5
    assert metrics["correct_classifications"] == 5
    assert metrics["total_classifications"] == 10
    
    # Test with all correct classifications
    metrics = agent.get_performance_metrics(10, 10)
    assert metrics["accuracy"] == 1.0

def test_matheus_agent_mcc_loading():
    """Test if Matheus agent correctly loads MCC data from CSV."""
    from mcc_classifier.agents.matheus_agent import MatheusAgent
    import os
    from pathlib import Path
    
    agent = MatheusAgent()
    
    # Check that we have MCC data loaded
    assert len(agent.mcc_data) > 0, "MCC data should be loaded"
    assert len(agent.mcc_risk_levels) > 0, "MCC risk levels should be loaded"
    
    # Check a few specific MCCs to verify correct loading
    # These are common MCCs that should be in the dataset
    assert "5812" in agent.mcc_data, "Restaurant MCC (5812) should be in the data"
    assert "7011" in agent.mcc_data, "Hotels MCC (7011) should be in the data"
    
    # Check that risk levels are properly loaded
    assert "5933" in agent.mcc_risk_levels, "Pawn shop MCC (5933) should have a risk level"
    
    # Verify a specific risk level
    if Path(agent.__file__).parent.joinpath("mcc_list.csv").exists():
        # This should be "prohibited" if loaded from the CSV
        assert agent.mcc_risk_levels.get("5933").lower() == "prohibited", "Pawn shops should be prohibited" 