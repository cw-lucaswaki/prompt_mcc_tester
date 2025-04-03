# MCC Tester

This repository contains tools and data for evaluating merchant category code (MCC) classification accuracy. It compares different MCC classification approaches against actual merchant data.

## Overview

The MCC Tester allows you to evaluate how accurately different classification methods can assign the correct Merchant Category Codes to businesses based on their names and legal names.

## Repository Structure

- `mcc_classifier/`: Core package containing the classification agents and utilities
- `tests/`: Unit tests for the classification agents
- `output/`: Results of classification runs in CSV format
- `data/`: Input data for testing
- `logs/`: Log files from test runs

## Key Files

- `run_comparison_test.py`: Script to run comparison tests across different classification agents
- `run_custom_mcc_evaluation.py`: Script to run custom MCC evaluations
- `run_waki_comparison.py`: Script to evaluate Waki's classification agent
- `test_matheus_agent.py`: Tests for Matheus's classification agent
- `test_waki_agent.py`: Tests for Waki's classification agent
- `requirements.txt`: Python package dependencies

## Results

The most recent comparison test (`output/custom_test_merchants_comparison_20250403_101504.csv`) shows accuracy rates of:
- Rafa: 68.00%
- Matheus: 14.00%
- Waki: 61.00%

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install the package in development mode: `pip install -e .`

## Running Tests

To run a comparison test:

```bash
python run_comparison_test.py
```

## License

[Specify license information] 