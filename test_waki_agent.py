import os
import argparse
import logging
import json
from mcc_classifier.agents.waki_agent import WakiAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_single_merchant(agent, merchant_name, legal_name=None):
    """
    Test the agent with a single merchant name.
    
    Args:
        agent: The agent to test
        merchant_name: The merchant name to classify
        legal_name: The legal name of the merchant (optional)
    """
    print(f"\n===== Classifying: '{merchant_name}' =====")
    if legal_name:
        print(f"Legal Name: '{legal_name}'")
        
    try:
        result = agent.classify(merchant_name, legal_name)
        
        print(f"\nResult:")
        print(f"MCC Code: {result['mcc_code']}")
        print(f"Description: {result['mcc_description']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        if 'analysis' in result:
            print(f"\nAnalysis: {result['analysis']}")
        
        if 'alternative_mccs' in result and result['alternative_mccs']:
            print("\nAlternative MCCs:")
            for alt in result['alternative_mccs']:
                print(f"- {alt['mcc_code']}: {alt['mcc_description']} (Confidence: {alt['confidence']:.2f})")
                
    except Exception as e:
        print(f"Error during classification: {str(e)}")

def test_multiple_merchants(agent, test_cases):
    """
    Test the agent with multiple test cases.
    
    Args:
        agent: The agent to test
        test_cases: List of dictionaries containing test cases
    """
    for case in test_cases:
        merchant_name = case.get('merchant_name', '')
        legal_name = case.get('legal_name', None)
        test_single_merchant(agent, merchant_name, legal_name)
        print("\n" + "-"*50)

def main():
    parser = argparse.ArgumentParser(description="Test the Waki MCC classification agent")
    parser.add_argument("--merchant", type=str, help="A single merchant name to classify")
    parser.add_argument("--legal-name", type=str, help="The legal name for the merchant")
    args = parser.parse_args()
    
    # Ensure OpenAI API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set. Using fallback classification.")
    
    # Create the agent
    agent = WakiAgent()
    
    if args.merchant:
        # Test a single merchant provided as argument
        test_single_merchant(agent, args.merchant, args.legal_name)
    else:
        # Test with a set of predefined test cases
        test_cases = [
            {'merchant_name': 'City Grocery'},
            {'merchant_name': 'Elite Electronics'},
            {'merchant_name': 'Joe\'s Pizza'},
            {'merchant_name': 'ABC Hardware'},
            {'merchant_name': 'Sunny Day Cafe'},
            {'merchant_name': 'Quick Mart'},
            {'merchant_name': 'The Bike Shop'},
            {'merchant_name': 'Maria\'s Salon'},
            {'merchant_name': 'Downtown Dental Clinic'},
            {'merchant_name': 'Pet Paradise'},
            {'merchant_name': 'The Fitness Center', 'legal_name': 'Health & Wellness LLC'},
            {'merchant_name': 'John Smith', 'legal_name': 'John Smith Consulting LLC'},
            {'merchant_name': 'Tech Solutions'},
            {'merchant_name': 'Sarah\'s Bakery'},
            {'merchant_name': 'Green Thumb Landscaping'}
        ]
        test_multiple_merchants(agent, test_cases)

if __name__ == "__main__":
    main() 