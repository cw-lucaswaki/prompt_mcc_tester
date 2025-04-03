from mcc_classifier.agents.base_agent import MCCClassifierAgent
import os
import csv
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. WakiAgent will use fallback classification.")

class WakiAgent(MCCClassifierAgent):
    """
    Waki's implementation of the MCC classifier agent.
    
    This agent uses GPT model with a carefully crafted prompt to classify merchants with their
    appropriate MCC codes. It loads the full MCC list from CSV for comprehensive classification.
    """
    
    def __init__(self):
        """Initialize the Waki MCC classifier agent."""
        super().__init__("Waki")
        
        # Load MCC data from CSV
        self.mcc_data = self._load_mcc_data()
        
        # Try to get API key from environment
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.warning("OpenAI API key not found in environment. Using fallback classification.")
        
        # Initialize OpenAI client if available
        self.client = None
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.logger.info("OpenAI client initialized successfully.")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {str(e)}")
    
    def _load_mcc_data(self) -> Dict[str, Dict[str, str]]:
        """
        Load MCC data from CSV file.
        
        Returns:
            Dict: Dictionary of MCC codes with their descriptions and classification status
        """
        mcc_dict = {}
        try:
            # First try to load from agents directory
            script_dir = Path(__file__).parent
            mcc_file = script_dir / "mcc_list.csv"
            
            self.logger.info(f"Loading MCC data from {mcc_file}")
            
            # Read CSV using pandas
            df = pd.read_csv(mcc_file)
            
            # Convert to dictionary structure
            for _, row in df.iterrows():
                try:
                    # Ensure MCC is treated as string and zero-padded to 4 digits
                    mcc = str(row['mcc']).strip().zfill(4)
                    
                    # Store the MCC data
                    mcc_dict[mcc] = {
                        'description': row['description'],
                        'classification': row['classification'] if 'classification' in row else 'Unknown'
                    }
                except Exception as e:
                    self.logger.warning(f"Error processing MCC row: {str(e)}")
                    continue
            
            self.logger.info(f"Loaded {len(mcc_dict)} MCC codes from CSV file")
            
            # If we have fewer than 100 MCCs, something went wrong with loading
            if len(mcc_dict) < 100:
                raise ValueError(f"Only loaded {len(mcc_dict)} MCCs, which seems too low")
                
        except Exception as e:
            self.logger.error(f"Error loading MCC data: {str(e)}")
            # Provide some fallback MCCs
            mcc_dict = {
                "5411": {"description": "Grocery Stores, Supermarkets", "classification": "Approved"},
                "5814": {"description": "Fast Food Restaurants", "classification": "Approved"},
                "5812": {"description": "Eating places and Restaurants", "classification": "Approved"},
                "5999": {"description": "Miscellaneous and Specialty Retail Stores", "classification": "Approved"},
                "7299": {"description": "Miscellaneous Personal Services", "classification": "Approved"},
                "5732": {"description": "Electronics Sales", "classification": "Approved"},
                "5045": {"description": "Computers, Computer Peripheral Equipment, Software", "classification": "Approved"},
                "5311": {"description": "Department Stores", "classification": "Approved"},
                "5200": {"description": "Home Supply Warehouse Stores", "classification": "Approved"},
                "5499": {"description": "Misc. Food Stores – Convenience Stores and Specialty Markets", "classification": "Approved"},
                "5651": {"description": "Family Clothing Stores", "classification": "Approved"},
                "8011": {"description": "Doctors and Physicians", "classification": "Approved"},
                "7230": {"description": "Barber and Beauty Shops", "classification": "Approved"},
                "7011": {"description": "Lodging – Hotels, Motels, Resorts", "classification": "Approved"}
            }
            self.logger.warning("Using fallback MCC data with limited entries")
        
        return mcc_dict
    
    def _create_prompt(self, merchant_data: Dict[str, Any]) -> str:
        """
        Create the prompt for the GPT API with effective classification instructions.
        
        Args:
            merchant_data: The full merchant record including name, legal name, and other fields
            
        Returns:
            The formatted prompt string
        """
        # Extract merchant info
        merchant_name = merchant_data.get('merchant_name', '')
        legal_name = merchant_data.get('legal_name', '')
        original_mcc = merchant_data.get('original_mcc_code', '')
        mcc_description = merchant_data.get('mcc_description', '')
        ai_original_description = merchant_data.get('ai_original_description', '')
        
        # Create a filtered list of MCCs to include in the prompt
        # Including all would make the prompt too large, so we'll select the most common/relevant ones
        common_mccs = [
            "5411", "5814", "5812", "5999", "5732", "5045", "7299", "5311", "5511",
            "5200", "5712", "5947", "5499", "5992", "5942", "8011", "8021", "8099",
            "7230", "4121", "7011", "4899", "5941", "7992", "5651", "5699", "5399",
            "5944", "5661", "5722", "5945", "0742", "7542", "7298", "7349", "8351",
            "1520", "7538", "4214", "5251", "5921"
        ]
        
        # Include original MCC if available
        if original_mcc and original_mcc not in common_mccs:
            common_mccs.append(original_mcc)
        
        # Format the selected MCCs as a JSON dictionary for easier access
        mcc_dict = {}
        for mcc in common_mccs:
            if mcc in self.mcc_data:
                mcc_dict[mcc] = self.mcc_data[mcc]['description']
        
        # Convert the dictionary to a formatted string
        mcc_examples_str = json.dumps(mcc_dict, indent=2)
        
        # Build the prompt
        prompt = f"""
# MCC Classification Task - PRECISION CRITICAL

## Context
Merchant Category Codes (MCCs) are critical for proper transaction categorization, regulatory compliance, and fee structures. Accurate classification directly impacts financial operations and reporting quality.

## Strict Accuracy Directives
1. Your classification MUST match standard industry assignments - this is NOT a creative task
2. You will be evaluated on exact matching to true MCC codes, NOT general categorization
3. Over-reliance on generic/catch-all categories (5999, 7299) is considered a FAILURE
4. Prioritize SPECIFIC industry codes over general ones (Example: code 5977 "Cosmetics" is better than 5999 "Misc Retail" for a cosmetics store)

## Merchant Information
Merchant Name: "{merchant_name}"
"""
        
        if legal_name and legal_name != merchant_name:
            prompt += f'Legal Name: "{legal_name}"\n'
        
        # Add original MCC information if available
        if original_mcc and mcc_description:
            prompt += f"""
Original MCC: {original_mcc} - {mcc_description}
"""
        
        # Add AI description if available
        if ai_original_description:
            prompt += f"""
AI Description: {ai_original_description}
"""
        
        prompt += f"""
## Expert Analysis Guidelines 
1. Examine merchant name for DIRECT business identifiers (exact matches like "restaurant," "salon")
2. Consider STANDARD INDUSTRY NAMING PATTERNS (e.g., "& Sons" often indicates family business, not literally sons)
3. For ambiguous names, apply STATISTICAL LIKELIHOOD (not creative interpretation)
4. Search for SPECIFIC business type indicators - most businesses have hints in their names
5. When business name contains a person's name followed by a service ("Smith Consulting"), focus on the service part
6. Names ending in "LLC", "Inc", etc. should be ignored for classification purposes

## Classification Hierarchies
- Level 1 (Best): PRECISE activity (e.g., 5977 "Cosmetics Stores" for a makeup shop)
- Level 2: Specific business category (e.g., 5651 "Family Clothing Stores" for general apparel)
- Level 3 (Avoid if possible): General category (e.g., 5999 "Misc. Retail" - USE ONLY AS LAST RESORT)

## Common Classification Patterns
- "Cafe", "Coffee", "Restaurant", "Grill", "Kitchen" → 5812/5814 (Restaurants/Fast Food)
- "Market", "Grocery", "Foods", "Supermarket" → 5411 (Grocery Stores)
- "Salon", "Hair", "Nails", "Barber" → 7230 (Beauty Shops)
- "Dr.", "Doctor", "Medical", "Health" → 8011/8099 (Medical Services)
- "Auto", "Car", "Repair", "Service" → 7538 (Auto Service Shops)
- "Law", "Legal", "Attorney" → 8111 (Legal Services)

## Critical Requirements
- NEVER default to general categories (5999: Misc. Retail or 7299: Misc. Services) unless absolutely no other classification applies
- If choosing between multiple plausible options, select the one most commonly assigned in the payments industry
- The TRUE MCC is almost NEVER a generic category if business activities can be determined

## Reference MCCs
{mcc_examples_str}

## Output Format
1. **Analysis**: Step-by-step business identification process
2. **Industry Classification**: Broad sector determination
3. **Primary MCC**: [MCC code]
4. **MCC Description**: [Full description]
5. **Reasoning**: Clear justification linking merchant name to industry standards
6. **Confidence**: [High/Medium/Low] with percentage (e.g., "85% confident")
7. **Alternative MCCs**: 2-3 other possible classifications in descending order with brief explanations

IMPORTANT: Remember this is a PRECISION task - your goal is to match the TRUE industry MCC assignment, not to make a reasonable guess.
"""
        
        return prompt
    
    def _parse_gpt_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the GPT response to extract the suggested MCC and related information.
        
        Args:
            response_text: The raw response from GPT
            
        Returns:
            Dictionary with parsed classification results
        """
        result = {
            'mcc_code': None,  # We'll set a default later if needed
            'mcc_description': "",
            'confidence': 0.7,
            'alternative_mccs': [],
            'analysis': "",
            'industry_classification': "",
            'reasoning': ""
        }
        
        try:
            # Parse the response
            lines = response_text.strip().split('\n')
            alternative_mccs = []
            confidence_value = 0.7  # Default
            
            # Flag to track when we're in the Alternative MCCs section
            in_alternatives = False
            current_alt = {}
            alt_explanation = ""
            
            # Pattern to capture MCC code with validation
            mcc_pattern = r'(\d{4})'
            
            # First pass - extract the main components
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Handle empty lines
                if not line:
                    continue
                
                # Extract analysis
                if re.search(r'analysis|step[- ]by[- ]step', line.lower(), re.IGNORECASE):
                    # Get the text from the next line(s) until we hit another section
                    analysis = []
                    j = i + 1
                    while j < len(lines) and not any(header in lines[j].lower() for header in 
                                                ["industry classification", "primary mcc", "mcc description", "**industry"]):
                        if lines[j].strip():
                            analysis.append(lines[j].strip())
                        j += 1
                    result['analysis'] = " ".join(analysis)
                
                # Extract industry classification
                elif "industry classification" in line.lower():
                    # Get the text from the next line(s) until we hit another section
                    classification = []
                    j = i + 1
                    while j < len(lines) and not any(header in lines[j].lower() for header in 
                                                ["primary mcc", "mcc description", "reasoning"]):
                        if lines[j].strip():
                            classification.append(lines[j].strip())
                        j += 1
                    result['industry_classification'] = " ".join(classification)
                
                # Extract primary MCC - look for MCC patterns more aggressively
                elif ("primary mcc" in line.lower() or 
                      "mcc code" in line.lower() or 
                      "mcc:" in line.lower()):
                    mcc_matches = re.findall(mcc_pattern, line)
                    if mcc_matches:
                        result['mcc_code'] = mcc_matches[0]
                
                # Extract MCC description
                elif "mcc description" in line.lower() or "description:" in line.lower():
                    # Get the description from the current line or next line
                    desc = re.sub(r'^.*?description:?\s*', '', line, flags=re.IGNORECASE).strip()
                    if not desc and i + 1 < len(lines):
                        desc = lines[i + 1].strip()
                    if desc:
                        result['mcc_description'] = desc
                
                # Extract reasoning
                elif "reasoning" in line.lower():
                    # Get the text from the next line(s) until we hit another section
                    reasoning = []
                    j = i + 1
                    while j < len(lines) and not any(header in lines[j].lower() for header in 
                                                ["confidence", "alternative mccs"]):
                        if lines[j].strip():
                            reasoning.append(lines[j].strip())
                        j += 1
                    result['reasoning'] = " ".join(reasoning)
                
                # Extract confidence
                elif "confidence" in line.lower():
                    confidence_line = line.lower()
                    if "high" in confidence_line:
                        confidence_value = 0.95
                    elif "medium" in confidence_line:
                        confidence_value = 0.8
                    elif "low" in confidence_line:
                        confidence_value = 0.6
                    
                    # Look for percentage values with variations
                    percentage_match = re.search(r'(\d{1,3})%|(\d{1,3})\s*percent', confidence_line)
                    if percentage_match:
                        percentage = int(percentage_match.group(1) or percentage_match.group(2))
                        confidence_value = min(0.99, max(0.01, percentage / 100))
                    
                    # Look for decimal confidence
                    decimal_match = re.search(r'confidence:?\s*(0\.\d+)', confidence_line)
                    if decimal_match:
                        try:
                            confidence_value = float(decimal_match.group(1))
                        except ValueError:
                            pass
                
                # Extract alternative MCCs
                elif "alternative mcc" in line.lower():
                    in_alternatives = True
                    continue
                
                # Process alternative MCCs
                elif in_alternatives:
                    # First check if this might be the start of a new alternative
                    mcc_matches = re.findall(mcc_pattern, line)
                    
                    if mcc_matches:
                        # If we've already started an alternative and now found a new one,
                        # save the previous one first
                        if current_alt and 'mcc_code' in current_alt:
                            if alt_explanation:
                                current_alt['explanation'] = alt_explanation
                            alternative_mccs.append(current_alt)
                            current_alt = {}
                            alt_explanation = ""
                        
                        # Extract the MCC code
                        alt_mcc = mcc_matches[0]
                        
                        # Extract the description if available
                        desc_match = re.search(r'\d{4}[:\s-]+([^\n]+)', line)
                        alt_desc = desc_match.group(1).strip() if desc_match else ""
                        
                        # Calculate confidence for alternatives (decreasing by position)
                        alt_confidence = max(0.1, confidence_value - 0.2 - (0.1 * len(alternative_mccs)))
                        
                        current_alt = {
                            'mcc_code': alt_mcc,
                            'mcc_description': alt_desc,
                            'confidence': round(alt_confidence, 2)
                        }
                    elif current_alt and 'mcc_code' in current_alt:
                        # This line likely contains explanation for the current alternative
                        if alt_explanation:
                            alt_explanation += " " + line
                        else:
                            alt_explanation = line
            
            # Don't forget to add the last alternative if there is one
            if current_alt and 'mcc_code' in current_alt:
                if alt_explanation:
                    current_alt['explanation'] = alt_explanation
                alternative_mccs.append(current_alt)
            
            # If we didn't find a primary MCC in the structured format, search more aggressively
            if not result['mcc_code']:
                # First try to extract from reasoning/analysis sections
                for section in [result.get('reasoning', ''), result.get('analysis', '')]:
                    if section:
                        mcc_matches = re.findall(r'MCC:?\s*(\d{4})|code:?\s*(\d{4})|suggested:?\s*(\d{4})', section, re.IGNORECASE)
                        if mcc_matches:
                            for match in mcc_matches:
                                for group in match:
                                    if group:
                                        result['mcc_code'] = group
                                        break
                                if result['mcc_code']:
                                    break
                
                # If still not found, look through all lines for any 4-digit codes
                if not result['mcc_code']:
                    for line in lines:
                        if "mcc" in line.lower() or "code" in line.lower() or "category" in line.lower():
                            mcc_matches = re.findall(mcc_pattern, line)
                            if mcc_matches:
                                result['mcc_code'] = mcc_matches[0]
                                # Try to extract description if available
                                desc_match = re.search(rf'{result["mcc_code"]}[:\s-]+([^\.]+)', line)
                                if desc_match:
                                    result['mcc_description'] = desc_match.group(1).strip()
                                break
            
            # If we still don't have an MCC code, set the default
            if not result['mcc_code']:
                result['mcc_code'] = "7299"  # Default if parsing fails
                result['mcc_description'] = "Miscellaneous Personal Services"
                self.logger.warning("Failed to extract MCC code from GPT response, using default 7299")
            
            # Save the confidence and alternative MCCs
            result['confidence'] = confidence_value
            result['alternative_mccs'] = alternative_mccs
            
            # Validate against known MCC codes and update description if needed
            if result['mcc_code'] in self.mcc_data:
                result['mcc_description'] = self.mcc_data[result['mcc_code']]['description']
            
            # If we don't have a reasoning but have an analysis, use that
            if not result['reasoning'] and result['analysis']:
                result['reasoning'] = f"Classification based on merchant name analysis: {result['analysis']}"
            elif not result['reasoning']:
                result['reasoning'] = "Classification based on merchant name patterns."
                
            # If we don't have an industry classification, derive it from the MCC
            if not result['industry_classification']:
                result['industry_classification'] = self._determine_industry(result['mcc_code'])
            
            # Fill any missing alternative MCCs to ensure we have at least 2
            if len(alternative_mccs) < 2:
                # Create different fallback alternatives based on the primary MCC
                # to avoid always defaulting to the same ones
                fallback_alts = []
                
                # Add different alternatives based on the general category
                if result['mcc_code'].startswith('5'):  # Retail/merchants
                    fallback_alts = [
                        {
                            'mcc_code': '5999', 
                            'mcc_description': 'Miscellaneous and Specialty Retail Stores', 
                            'confidence': 0.3,
                            'explanation': 'General retail fallback option'
                        },
                        {
                            'mcc_code': '5499', 
                            'mcc_description': 'Misc. Food Stores – Convenience Stores and Specialty Markets', 
                            'confidence': 0.25,
                            'explanation': 'Food retail alternative if applicable'
                        }
                    ]
                elif result['mcc_code'].startswith('7'):  # Services
                    fallback_alts = [
                        {
                            'mcc_code': '7399', 
                            'mcc_description': 'Business Services, Not Elsewhere Classified', 
                            'confidence': 0.3,
                            'explanation': 'Business services alternative'
                        },
                        {
                            'mcc_code': '7311', 
                            'mcc_description': 'Advertising Services', 
                            'confidence': 0.25,
                            'explanation': 'Marketing/promotion services if applicable'
                        }
                    ]
                elif result['mcc_code'].startswith('8'):  # Professional services
                    fallback_alts = [
                        {
                            'mcc_code': '8999', 
                            'mcc_description': 'Professional Services, Not Elsewhere Classified', 
                            'confidence': 0.3,
                            'explanation': 'Alternative professional services classification'
                        },
                        {
                            'mcc_code': '8931', 
                            'mcc_description': 'Accounting, Auditing, and Bookkeeping Services', 
                            'confidence': 0.25,
                            'explanation': 'Financial professional services if applicable'
                        }
                    ]
                else:
                    fallback_alts = [
                        {
                            'mcc_code': '5999', 
                            'mcc_description': 'Miscellaneous and Specialty Retail Stores', 
                            'confidence': 0.3,
                            'explanation': 'Retail alternative'
                        },
                        {
                            'mcc_code': '7299', 
                            'mcc_description': 'Miscellaneous Personal Services', 
                            'confidence': 0.25,
                            'explanation': 'Services alternative'
                        }
                    ]
                
                for alt in fallback_alts:
                    if len(alternative_mccs) >= 2:
                        break
                    if alt['mcc_code'] != result['mcc_code']:
                        alternative_mccs.append(alt)
                
                result['alternative_mccs'] = alternative_mccs
            
        except Exception as e:
            self.logger.error(f"Error parsing GPT response: {str(e)}", exc_info=True)
            # Set defaults if we encounter an error
            if not result['mcc_code']:
                result['mcc_code'] = "7299"
                result['mcc_description'] = "Miscellaneous Personal Services"
        
        return result
    
    def _fallback_classify(self, merchant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback classification method when GPT is not available.
        
        Args:
            merchant_data: The full merchant record 
            
        Returns:
            Dictionary with classification results
        """
        # Extract merchant info
        merchant_name = merchant_data.get('merchant_name', '')
        legal_name = merchant_data.get('legal_name', '')
        original_mcc = merchant_data.get('original_mcc_code', '')
        
        # If original MCC is provided and it's valid, use it with reduced confidence
        if original_mcc and original_mcc in self.mcc_data:
            return {
                'mcc_code': original_mcc,
                'mcc_description': self.mcc_data[original_mcc]['description'],
                'confidence': 0.7,
                'alternative_mccs': [
                    {'mcc_code': '5999', 'mcc_description': 'Miscellaneous and Specialty Retail Stores', 'confidence': 0.3},
                    {'mcc_code': '7299', 'mcc_description': 'Miscellaneous Personal Services', 'confidence': 0.2}
                ],
                'industry_classification': self._determine_industry(original_mcc),
                'analysis': 'Using provided original MCC code',
                'reasoning': 'Classification based on original MCC code'
            }
        
        # Simple keyword-based classification
        merchant_name_lower = merchant_name.lower()
        legal_name_lower = legal_name.lower() if legal_name else ''
        
        # Combine both names for classification
        combined_text = f"{merchant_name_lower} {legal_name_lower}"
        
        # Pattern matching for common business types
        pattern_matches = []
        
        # Food and dining patterns
        if re.search(r'\b(restaurant|cafe|café|coffee|bistro|pizzeria|grill|diner|eatery|food)\b', combined_text):
            pattern_matches.append(('5812', 'Eating places and Restaurants', 0.9))
        if re.search(r'\b(fast food|burger|taco|sandwich|sub|quick)\b', combined_text):
            pattern_matches.append(('5814', 'Fast Food Restaurants', 0.9))
        if re.search(r'\b(grocery|market|supermarket|food store|supermercado|groceries)\b', combined_text):
            pattern_matches.append(('5411', 'Grocery Stores, Supermarkets', 0.9))
        if re.search(r'\b(convenience|mini mart|7[\s-]?eleven|corner store)\b', combined_text):
            pattern_matches.append(('5499', 'Misc. Food Stores – Convenience Stores and Specialty Markets', 0.85))
        
        # Retail patterns
        if re.search(r'\b(electronics|computer|laptop|phone|mobile|tech|digital)\b', combined_text):
            pattern_matches.append(('5732', 'Electronics Sales', 0.85))
        if re.search(r'\b(clothing|apparel|fashion|wear|dress|outfit|garment)\b', combined_text):
            pattern_matches.append(('5651', 'Family Clothing Stores', 0.85))
        if re.search(r'\b(hardware|tool|diy|home improvement)\b', combined_text):
            pattern_matches.append(('5251', 'Hardware Stores', 0.85))
        if re.search(r'\b(furniture|home furnishing|mattress|bed|sofa|couch)\b', combined_text):
            pattern_matches.append(('5712', 'Furniture, Home Furnishings, and Equipment Stores, Except Appliances', 0.85))
        if re.search(r'\b(department store|general store|variety|walmart|target)\b', combined_text):
            pattern_matches.append(('5311', 'Department Stores', 0.85))
        if re.search(r'\b(shoe|footwear|sneaker|boot)\b', combined_text):
            pattern_matches.append(('5661', 'Shoe Stores', 0.9))
        
        # Services patterns
        if re.search(r'\b(salon|barber|hair|beauty|nail|spa)\b', combined_text):
            pattern_matches.append(('7230', 'Barber and Beauty Shops', 0.9))
        if re.search(r'\b(repair|fix|mend|service)\b', combined_text):
            pattern_matches.append(('7699', 'Repair Shops and Related Services –Miscellaneous', 0.8))
        if re.search(r'\b(auto|car|vehicle|mechanic|automotive)\b', combined_text):
            pattern_matches.append(('7538', 'Automotive Service Shops', 0.9))
        if re.search(r'\b(hotel|motel|inn|lodging|accommodation)\b', combined_text):
            pattern_matches.append(('7011', 'Lodging – Hotels, Motels, Resorts', 0.9))
        if re.search(r'\b(clean|cleaning|janitorial|maid|wash|laundry)\b', combined_text):
            pattern_matches.append(('7349', 'Cleaning and Maintenance, Janitorial Services', 0.85))
        
        # Professional services patterns
        if re.search(r'\b(doctor|physician|medical|clinic|health|healthcare)\b', combined_text):
            pattern_matches.append(('8011', 'Doctors and Physicians', 0.9))
        if re.search(r'\b(dentist|dental|orthodont|teeth|tooth)\b', combined_text):
            pattern_matches.append(('8021', 'Dentists and Orthodontists', 0.9))
        if re.search(r'\b(law|attorney|legal|lawyer|advocate)\b', combined_text):
            pattern_matches.append(('8111', 'Legal Services, Attorneys', 0.9))
        if re.search(r'\b(consult|consulting|advisor|counsel)\b', combined_text):
            pattern_matches.append(('7392', 'Management, Consulting, and Public Relations Services', 0.8))
        if re.search(r'\b(insurance|insure|policy|coverage)\b', combined_text):
            pattern_matches.append(('6300', 'Insurance Sales, Underwriting and Premiums', 0.85))
        
        # Specialty retail
        if re.search(r'\b(pet|animal|dog|cat|bird)\b', combined_text):
            pattern_matches.append(('5995', 'Pet Shops, Pet Foods, and Supplies Stores', 0.9))
        if re.search(r'\b(book|comic|magazine|publication)\b', combined_text):
            pattern_matches.append(('5942', 'Bookstores', 0.9))
        if re.search(r'\b(pharmacy|drug|prescription|medicine|pharmaceutical)\b', combined_text):
            pattern_matches.append(('5912', 'Drug Stores and Pharmacies', 0.9))
        if re.search(r'\b(toy|game|hobby|play)\b', combined_text):
            pattern_matches.append(('5945', 'Hobby, Toy, and Game Shops', 0.85))
        
        # If we have pattern matches, use the one with highest confidence
        if pattern_matches:
            # Sort by confidence (highest first)
            pattern_matches.sort(key=lambda x: x[2], reverse=True)
            top_match = pattern_matches[0]
            
            # Create alternative MCCs from other pattern matches
            alternative_mccs = []
            for mcc, desc, conf in pattern_matches[1:3]:  # Take up to 2 alternatives
                alternative_mccs.append({
                    'mcc_code': mcc,
                    'mcc_description': desc,
                    'confidence': conf - 0.2  # Reduce confidence for alternatives
                })
            
            # Add generic fallbacks if needed
            while len(alternative_mccs) < 2:
                if top_match[0] != '5999':
                    alternative_mccs.append({
                        'mcc_code': '5999',
                        'mcc_description': 'Miscellaneous and Specialty Retail Stores',
                        'confidence': 0.3
                    })
                    break
                if top_match[0] != '7299':
                    alternative_mccs.append({
                        'mcc_code': '7299',
                        'mcc_description': 'Miscellaneous Personal Services',
                        'confidence': 0.3
                    })
                    break
            
            return {
                'mcc_code': top_match[0],
                'mcc_description': top_match[1],
                'confidence': top_match[2],
                'alternative_mccs': alternative_mccs,
                'industry_classification': self._determine_industry(top_match[0]),
                'analysis': f'Identified business type from name patterns',
                'reasoning': f'Pattern matching identified keywords related to {top_match[1]}'
            }
        
        # Common business types with their MCCs - fallback to simple keyword matching
        keyword_map = {
            'restaurant': {'mcc': '5812', 'desc': 'Eating places and Restaurants'},
            'café': {'mcc': '5812', 'desc': 'Eating places and Restaurants'},
            'cafe': {'mcc': '5812', 'desc': 'Eating places and Restaurants'},
            'coffee': {'mcc': '5812', 'desc': 'Eating places and Restaurants'},
            'pizza': {'mcc': '5812', 'desc': 'Eating places and Restaurants'},
            'food': {'mcc': '5499', 'desc': 'Misc. Food Stores – Convenience Stores and Specialty Markets'},
            'grocery': {'mcc': '5411', 'desc': 'Grocery Stores, Supermarkets'},
            'market': {'mcc': '5411', 'desc': 'Grocery Stores, Supermarkets'},
            'supermarket': {'mcc': '5411', 'desc': 'Grocery Stores, Supermarkets'},
            'electronics': {'mcc': '5732', 'desc': 'Electronics Sales'},
            'computer': {'mcc': '5045', 'desc': 'Computers, Computer Peripheral Equipment, Software'},
            'software': {'mcc': '5045', 'desc': 'Computers, Computer Peripheral Equipment, Software'},
            'hardware': {'mcc': '5251', 'desc': 'Hardware Stores'},
            'clothing': {'mcc': '5651', 'desc': 'Family Clothing Stores'},
            'apparel': {'mcc': '5699', 'desc': 'Miscellaneous Apparel and Accessory Shops'},
            'fashion': {'mcc': '5699', 'desc': 'Miscellaneous Apparel and Accessory Shops'},
            'shoe': {'mcc': '5661', 'desc': 'Shoe Stores'},
            'footwear': {'mcc': '5661', 'desc': 'Shoe Stores'},
            'salon': {'mcc': '7230', 'desc': 'Barber and Beauty Shops'},
            'spa': {'mcc': '7298', 'desc': 'Health and Beauty Shops'},
            'beauty': {'mcc': '7298', 'desc': 'Health and Beauty Shops'},
            'barber': {'mcc': '7230', 'desc': 'Barber and Beauty Shops'},
            'hair': {'mcc': '7230', 'desc': 'Barber and Beauty Shops'},
            'doctor': {'mcc': '8011', 'desc': 'Doctors and Physicians'},
            'medical': {'mcc': '8099', 'desc': 'Medical Services and Health Practitioners'},
            'dentist': {'mcc': '8021', 'desc': 'Dentists and Orthodontists'},
            'dental': {'mcc': '8021', 'desc': 'Dentists and Orthodontists'},
            'clean': {'mcc': '7349', 'desc': 'Cleaning and Maintenance, Janitorial Services'},
            'repair': {'mcc': '7699', 'desc': 'Repair Shops and Related Services –Miscellaneous'},
            'auto': {'mcc': '7538', 'desc': 'Automotive Service Shops'},
            'car': {'mcc': '7538', 'desc': 'Automotive Service Shops'},
            'hotel': {'mcc': '7011', 'desc': 'Lodging – Hotels, Motels, Resorts'},
            'motel': {'mcc': '7011', 'desc': 'Lodging – Hotels, Motels, Resorts'},
            'pet': {'mcc': '5995', 'desc': 'Pet Shops, Pet Foods, and Supplies Stores'},
            'toy': {'mcc': '5945', 'desc': 'Hobby, Toy, and Game Shops'}
        }
        
        # Look for keywords in merchant name
        for keyword, mcc_info in keyword_map.items():
            if keyword in combined_text:
                return {
                    'mcc_code': mcc_info['mcc'],
                    'mcc_description': mcc_info['desc'],
                    'confidence': 0.7,
                    'alternative_mccs': [
                        {'mcc_code': '5999', 'mcc_description': 'Miscellaneous and Specialty Retail Stores', 'confidence': 0.3},
                        {'mcc_code': '7299', 'mcc_description': 'Miscellaneous Personal Services', 'confidence': 0.2}
                    ],
                    'industry_classification': self._determine_industry(mcc_info['mcc']),
                    'analysis': f'Found keyword "{keyword}" in merchant name',
                    'reasoning': f'Keyword matching identified "{keyword}" related to {mcc_info["desc"]}'
                }
        
        # Default fallback - try to use the first word of the merchant name to guess personal vs. business service
        if ' ' in merchant_name and len(merchant_name.split(' ')[0]) > 2:
            # If merchant name starts with what looks like a person's name, likely a service
            return {
                'mcc_code': '7299',
                'mcc_description': 'Miscellaneous Personal Services',
                'confidence': 0.6,
                'alternative_mccs': [
                    {'mcc_code': '5999', 'mcc_description': 'Miscellaneous and Specialty Retail Stores', 'confidence': 0.3},
                    {'mcc_code': '7399', 'mcc_description': 'Business Services', 'confidence': 0.2}
                ],
                'industry_classification': 'Services',
                'analysis': 'Unable to identify specific business type from name',
                'reasoning': 'Name suggests a personal or professional service business'
            }
        else:
            # Default to retail as slightly more common
            return {
                'mcc_code': '5999',
                'mcc_description': 'Miscellaneous and Specialty Retail Stores',
                'confidence': 0.6,
                'alternative_mccs': [
                    {'mcc_code': '7299', 'mcc_description': 'Miscellaneous Personal Services', 'confidence': 0.3},
                    {'mcc_code': '7399', 'mcc_description': 'Business Services', 'confidence': 0.2}
                ],
                'industry_classification': 'Retail/Merchants',
                'analysis': 'Unable to identify specific business type from name',
                'reasoning': 'Defaulting to retail classification based on statistical prevalence'
            }
    
    def classify(self, merchant_name, legal_name=None, **merchant_data):
        """
        Classify a merchant using Waki's algorithm.
        
        Args:
            merchant_name (str): The name of the merchant.
            legal_name (str, optional): The legal name of the merchant.
            **merchant_data: Additional merchant data fields that can include:
                - original_mcc_code: The original MCC code
                - mcc_description: Description of the original MCC
                - ai_original_description: AI's original description
                - And any other fields available in the merchant record
            
        Returns:
            dict: Classification result with suggested MCC and confidence.
        """
        self.logger.info(f"Classifying merchant: {merchant_name}")
        
        # Prepare full merchant data dictionary
        full_merchant_data = {
            'merchant_name': merchant_name,
            'legal_name': legal_name,
            **merchant_data
        }
        
        # Check if we can use GPT
        if not OPENAI_AVAILABLE or not self.client:
            self.logger.warning("OpenAI client not available. Using fallback classification.")
            return self._fallback_classify(full_merchant_data)
        
        try:
            # Create the prompt
            prompt = self._create_prompt(full_merchant_data)
            
            # Make the API call
            self.logger.info(f"Sending request to GPT for merchant: {merchant_name}")
            
            system_prompt = """You are an expert in merchant classification and Merchant Category Codes (MCCs) with extensive knowledge of global business types across all industries.

You specialize in EXACT, ACCURATE MCC assignment according to industry standards. This is NOT a creative task - proper classification requires:

1. Precise adherence to standard industry MCC assignments
2. Avoidance of generic categories (7299, 5999) whenever possible
3. Focus on direct business indicators in merchant names
4. Prioritizing industry-specific codes over general alternatives
5. Knowledge of common merchant naming patterns across sectors

ACCURACY DIRECTIVES:
- You will be evaluated on EXACT MCC CODE MATCHING
- You must use standard industry assignments, not logical or creative alternatives
- Generic categories should be used only when absolutely necessary
- Most merchants can be classified by specific indicators in their names
- Follow standard payment industry classification patterns, not your general knowledge

Your goal is to match the TRUE MCC assignment according to payment industry standards.
"""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Use GPT-4o for better results
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1  # Very low temperature for more consistent, deterministic results
                )
            except Exception as model_error:
                # If GPT-4o fails, try GPT-4 or GPT-3.5-turbo as fallback
                self.logger.warning(f"Failed to use GPT-4o, falling back to GPT-4: {str(model_error)}")
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4",  # Fallback to GPT-4
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                except Exception as fallback_error:
                    # If GPT-4 also fails, try GPT-3.5-turbo
                    self.logger.warning(f"Failed to use GPT-4, falling back to GPT-3.5-turbo: {str(fallback_error)}")
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Last resort fallback
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2  # Slightly higher for GPT-3.5 but still conservative
                    )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            self.logger.debug(f"GPT response: {response_text}")
            
            # Parse the response
            result = self._parse_gpt_response(response_text)
            
            # Log the classification result
            self.logger.info(f"Classified '{merchant_name}' as MCC {result['mcc_code']} - {result['mcc_description']} with confidence {result['confidence']}")
            
            return result
                
        except Exception as e:
            self.logger.error(f"Error classifying with GPT: {str(e)}", exc_info=True)
            self.logger.error(f"Request data: merchant_name='{merchant_name}', legal_name='{legal_name}'")
            return self._fallback_classify(full_merchant_data) 

    def _determine_industry(self, mcc: str) -> str:
        """
        Determine the broad industry classification based on MCC code.
        
        Args:
            mcc: The MCC code
            
        Returns:
            The industry classification as a string
        """
        if not mcc or len(mcc) < 2:
            return "Unknown"
            
        # First digit of MCC code generally indicates the industry
        first_digit = mcc[0]
        
        # Major industry classifications
        if first_digit == '5':
            # Further classify retail
            if mcc.startswith('54'):
                return "Food and Grocery Retail"
            elif mcc.startswith('56'):
                return "Apparel and Accessories Retail"
            elif mcc.startswith('57'):
                return "Home Furnishings and Electronics Retail"
            elif mcc.startswith('58'):
                return "Restaurants and Food Service"
            elif mcc.startswith('59'):
                return "Specialty Retail"
            else:
                return "Retail/Merchants"
                
        elif first_digit == '7':
            # Further classify services
            if mcc.startswith('70') or mcc.startswith('71'):
                return "Travel and Lodging"
            elif mcc.startswith('72'):
                return "Personal Services"
            elif mcc.startswith('73'):
                return "Business and Professional Services"
            elif mcc.startswith('74') or mcc.startswith('76'):
                return "Repair and Maintenance Services"
            elif mcc.startswith('75'):
                return "Auto Services"
            elif mcc.startswith('78') or mcc.startswith('79'):
                return "Entertainment and Recreation"
            else:
                return "Services"
                
        elif first_digit == '8':
            # Further classify professional services
            if mcc.startswith('80'):
                return "Healthcare"
            elif mcc.startswith('81'):
                return "Legal Services"
            elif mcc.startswith('82') or mcc.startswith('83'):
                return "Educational Services"
            elif mcc.startswith('86'):
                return "Membership Organizations"
            else:
                return "Professional Services"
                
        elif first_digit == '4':
            return "Transportation/Utilities"
        elif first_digit == '6':
            return "Financial Services"
        elif first_digit == '9':
            return "Government Services"
        elif first_digit in ['0', '1', '2', '3']:
            return "Contractors/Construction/Agriculture"
        else:
            return "Other Business Categories" 