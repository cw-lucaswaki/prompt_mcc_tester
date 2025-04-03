import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from mcc_classifier.evaluator import MCCEvaluator
from mcc_classifier.agents.base_agent import MCCClassifierAgent

@pytest.fixture
def mock_agents():
    """Create mock MCC classifier agents for testing."""
    # Create mock agents
    agent1 = MagicMock(spec=MCCClassifierAgent)
    agent1.name = "Agent1"
    agent1.classify.return_value = {
        'mcc_code': '5411',
        'mcc_description': 'Grocery Stores',
        'confidence': 0.9,
        'alternative_mccs': []
    }
    
    agent2 = MagicMock(spec=MCCClassifierAgent)
    agent2.name = "Agent2"
    agent2.classify.return_value = {
        'mcc_code': '5812',
        'mcc_description': 'Restaurants',
        'confidence': 0.8,
        'alternative_mccs': []
    }
    
    return [agent1, agent2]

@pytest.fixture
def sample_input_data():
    """Create sample input data for testing."""
    return [
        {
            "Merchant Name": "Merchant1",
            "Legal Name": "Legal1",
            "Actual MCC code": "5411",
            "MCC Description": "Grocery Stores"
        },
        {
            "Merchant Name": "Merchant2",
            "Legal Name": "Legal2",
            "Actual MCC code": "5812",
            "MCC Description": "Restaurants"
        }
    ]

@pytest.fixture
def temp_input_csv(tmp_path, sample_input_data):
    """Create a temporary input CSV file for testing."""
    input_path = os.path.join(tmp_path, "input.csv")
    df = pd.DataFrame(sample_input_data)
    df.to_csv(input_path, index=False)
    return input_path

@pytest.fixture
def temp_output_csv(tmp_path):
    """Create a temporary output CSV file path for testing."""
    return os.path.join(tmp_path, "output.csv")

def test_evaluator_initialization(mock_agents):
    """Test initializing the evaluator with agents."""
    evaluator = MCCEvaluator(mock_agents)
    assert evaluator.agents == mock_agents

@patch('mcc_classifier.agents.agent_factory.AgentFactory.create_all_agents')
def test_evaluator_initialization_default_agents(mock_create_all_agents, mock_agents):
    """Test initializing the evaluator with default agents."""
    mock_create_all_agents.return_value = mock_agents
    evaluator = MCCEvaluator()
    assert evaluator.agents == mock_agents

@patch('mcc_classifier.utils.data_handler.DataHandler.read_csv')
@patch('mcc_classifier.utils.data_handler.DataHandler.write_csv')
def test_evaluate(mock_write_csv, mock_read_csv, mock_agents, sample_input_data, temp_input_csv, temp_output_csv):
    """Test the evaluate method of the evaluator."""
    # Setup mock read_csv to return sample data
    mock_read_csv.return_value = sample_input_data
    
    # Create evaluator with mock agents
    evaluator = MCCEvaluator(mock_agents)
    
    # Run evaluation
    metrics = evaluator.evaluate(temp_input_csv, temp_output_csv)
    
    # Check that read_csv was called with the input path
    mock_read_csv.assert_called_once_with(temp_input_csv)
    
    # Check that write_csv was called with the output path
    mock_write_csv.assert_called_once()
    assert mock_write_csv.call_args[0][0] == temp_output_csv
    
    # Check that each agent's classify method was called for each merchant
    for agent in mock_agents:
        assert agent.classify.call_count == len(sample_input_data)
    
    # Check that metrics were calculated for each agent
    assert "Agent1" in metrics
    assert "Agent2" in metrics
    
    # Check that Agent1 got 1 correct (first merchant) and Agent2 got 1 correct (second merchant)
    assert metrics["Agent1"]["correct_classifications"] == 1
    assert metrics["Agent2"]["correct_classifications"] == 1
    assert metrics["Agent1"]["total_classifications"] == 2
    assert metrics["Agent2"]["total_classifications"] == 2
    assert metrics["Agent1"]["accuracy"] == 0.5
    assert metrics["Agent2"]["accuracy"] == 0.5

@patch('mcc_classifier.utils.data_handler.DataHandler.read_csv')
def test_evaluate_with_missing_data(mock_read_csv, mock_agents, temp_input_csv, temp_output_csv):
    """Test evaluation with missing data in the input."""
    # Input data with missing Merchant Name and Actual MCC code
    input_data = [
        {
            "Merchant Name": "",
            "Legal Name": "Legal1",
            "Actual MCC code": "5411",
            "MCC Description": "Grocery Stores"
        },
        {
            "Merchant Name": "Merchant2",
            "Legal Name": "Legal2",
            "Actual MCC code": "",
            "MCC Description": "Restaurants"
        },
        {
            "Merchant Name": "Merchant3",
            "Legal Name": "Legal3",
            "Actual MCC code": "5812",
            "MCC Description": "Restaurants"
        }
    ]
    
    # Setup mock read_csv to return input data with missing values
    mock_read_csv.return_value = input_data
    
    # Create evaluator with mock agents
    evaluator = MCCEvaluator(mock_agents)
    
    # Run evaluation
    metrics = evaluator.evaluate(temp_input_csv, temp_output_csv)
    
    # Check that each agent's classify method was called only for the valid merchant
    for agent in mock_agents:
        assert agent.classify.call_count == 1  # Only the third merchant is valid
    
    # Check that metrics were calculated correctly
    assert metrics["Agent1"]["total_classifications"] == 1
    assert metrics["Agent2"]["total_classifications"] == 1 