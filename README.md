# MCC Classification Evaluation Tool

A Python tool for evaluating the performance of different AI agents in classifying merchants with their correct Merchant Category Codes (MCCs).

## Overview

This tool compares the performance of three different AI agents (Rafa, Matheus, and Waki) in assigning MCCs to merchants based on their names. It takes a CSV file containing merchant data with actual MCCs as input and generates a CSV file with the MCC recommendations from each agent and performance metrics.

## Installation

### Prerequisites

- Python 3.6 or higher
- OpenAI API key (for Rafa's and Matheus's agents)
- Pydantic (for Matheus's agent)

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd mcc-classifier
   ```

2. Install the package and dependencies:
   ```
   pip install -e .
   ```

3. Set up the OpenAI API key (required for Rafa's and Matheus's agents):
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY=your_api_key_here

   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your_api_key_here"
   ```

## Usage

### Command-line Interface

To run the evaluation using the command-line interface:

```bash
./run_mcc_evaluation.py -i <input-csv-file> -o <output-csv-file>
```

Or using the console script:

```bash
mcc-evaluate -i <input-csv-file> -o <output-csv-file>
```

#### Command-line Options

- `-i, --input`: Path to the input CSV file (required)
- `-o, --output`: Path to the output CSV file (optional, default: `output/mcc_evaluation_<timestamp>.csv`)
- `-v, --verbose`: Enable verbose logging (optional)
- `--no-log-file`: Disable logging to file (optional)

### Input File Format

The input CSV file should contain the following columns:
- Merchant Name
- Legal Name
- Actual MCC code
- MCC Description

Example:
```csv
Merchant Name,Legal Name,Actual MCC code,MCC Description
Walmart,Walmart Inc.,5411,Grocery Stores
Amazon,Amazon.com Inc.,5964,Direct Marketing - Catalog Merchants
```

### Output File Format

The output CSV file will contain:
- Merchant Name
- Legal Name
- Actual MCC
- MCC Description
- Rafa's suggested MCC
- Matheus's suggested MCC
- Waki's suggested MCC
- Performance metrics for each agent

## Agent Implementations

### Rafa Agent

The Rafa agent uses OpenAI's GPT-4 model to classify merchants with the appropriate MCC codes. It requires an OpenAI API key to be set in the environment. If the API key is not available, it falls back to a simpler keyword-based classification approach.

#### Features

- Uses OpenAI's GPT-4 to analyze merchant names and suggest appropriate MCCs
- Analyzes both merchant name and legal name to make smarter classifications
- Provides confidence scores and alternative MCC suggestions
- Has a fallback mechanism when the API is not available or encounters an error

#### Configuration

To use the Rafa agent with OpenAI, make sure to:
1. Install the OpenAI package: `pip install openai>=1.0.0`
2. Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY=your_api_key_here`

### Matheus Agent

The Matheus agent uses OpenAI's models with a sophisticated three-tier approach to classify merchants. It evaluates merchants in multiple stages to determine the most appropriate MCC, with special attention to risk assessment.

#### Features

- Implements a three-tier classification approach:
  1. Initial classification with limited MCC list
  2. Risk-based classification for potentially high-risk merchants
  3. Full MCC database search for ambiguous cases
- Considers whether merchant names are non-descriptive (just the owner's name)
- Flags potentially high-risk businesses
- Detects suspicious classifications where the MCC seems intentionally misleading
- Prioritizes higher-risk categories for questionable businesses

#### Configuration

To use the Matheus agent with OpenAI, make sure to:
1. Install the OpenAI and Pydantic packages: `pip install openai>=1.0.0 pydantic>=2.0.0`
2. Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY=your_api_key_here`

### Waki Agent

The Waki agent uses a pattern matching approach to classify merchants based on their names.

## Adding New Agents

To add a new MCC classification agent:

1. Create a new agent class that inherits from `MCCClassifierAgent` in a new file in the `mcc_classifier/agents/` directory:

```python
from mcc_classifier.agents.base_agent import MCCClassifierAgent

class NewAgent(MCCClassifierAgent):
    def __init__(self):
        super().__init__("New Agent Name")
    
    def classify(self, merchant_name, legal_name=None):
        # Implement your classification logic here
        # ...
        
        return {
            'mcc_code': mcc_code,
            'mcc_description': mcc_description,
            'confidence': confidence,
            'alternative_mccs': alternative_mccs
        }
```

2. Update the `AgentFactory` class in `mcc_classifier/agents/agent_factory.py` to include your new agent:

```python
def create_agent(agent_type):
    # ...
    elif agent_type == "new_agent":
        logger.info("Creating New Agent")
        return NewAgent()
    # ...

def create_all_agents():
    # ...
    return [
        RafaAgent(),
        MatheusAgent(),
        WakiAgent(),
        NewAgent()  # Add your new agent here
    ]
```

## License

[Specify license information] 