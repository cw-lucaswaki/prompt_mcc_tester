from mcc_classifier.agents.base_agent import MCCClassifierAgent
import re
import os
import logging
from typing import Dict, Any, List, Tuple

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. RafaAgent will use fallback classification.")

class RafaAgent(MCCClassifierAgent):
    """
    Rafa's implementation of the MCC classifier agent based on Rafael Pereira's approach.
    
    This agent uses OpenAI's GPT-4 to classify merchants with their appropriate MCC codes.
    """
    
    def __init__(self):
        """Initialize the Rafa MCC classifier agent."""
        super().__init__("Rafa")
        
        # Dictionary of common MCC codes and their descriptions
        self.mcc_data = {
            "0780": "Landscaping & Lawn Care",
            "1520": "General Contractors",
            "1711": "HVAC & Plumbing",
            "1731": "Electrical",
            "1740": "Masonry & Tile",
            "1750": "Carpentry",
            "1761": "Roofing & Siding",
            "1771": "Concrete",
            "1799": "Special Trade",
            "4789": "Transportation",
            "5211": "Building Materials",
            "5251": "Hardware",
            "5311": "Department Stores",
            "5399": "Other Retail",
            "5411": "Grocery Stores, Supermarkets",
            "5499": "Food & Convenience",
            "5533": "Auto Parts",
            "5541": "Gas & Fuel",
            "5651": "Apparel",
            "5661": "Footwear",
            "5699": "Clothing & Accessories",
            "5812": "Restaurants",
            "5814": "Fast Food",
            "5940": "Bike Shops",
            "5941": "Sporting Goods",
            "5942": "Bookstores",
            "5943": "Office & Stationery",
            "5945": "Hobbies & Toys",
            "5947": "Gifts & Souvenirs",
            "5970": "Arts & Crafts",
            "5977": "Cosmetics",
            "5992": "Florists",
            "5995": "Pet Supplies",
            "7011": "Hotels & Lodging",
            "7210": "Laundry & Cleaning",
            "7211": "Laundry Services–Family and Commercial",
            "7216": "Dry Cleaners",
            "7221": "Photography",
            "7230": "Salons & Barbers",
            "7251": "Shoe Repair & Shine",
            "7298": "Health and Beauty Spas",
            "7299": "Other Services",
            "7399": "Other B2B Services",
            "7542": "Car Wash",
            "7549": "Towing",
            "7699": "Repair Shops & Services",
            "7997": "Country Clubs & Private Golf Courses",
            "8099": "Medical & Health Services",
            "8299": "Educational Services",
            "5964": "Direct Marketing - Catalog Merchants",
            "5732": "Electronics Stores",
            "5912": "Drug Stores and Pharmacies",
            "5200": "Home Supply Warehouse Stores",
        }
        
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
    
    def _create_prompt(self, merchant_data: Dict[str, Any]) -> str:
        """
        Create the prompt for the OpenAI API.
        
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
        
        # Format the list of MCC codes for the prompt
        mcc_examples = "\n".join([f"- {code} : {desc}" for code, desc in self.mcc_data.items()])
        
        # Build the prompt
        prompt = f"""
        A merchant named "{merchant_name}" needs to be classified with an appropriate MCC code.
        """
        
        if legal_name and legal_name != merchant_name:
            prompt += f'The legal representative\'s name is "{legal_name}".\n'
        
        # Add original MCC information if available
        if original_mcc and mcc_description:
            prompt += f"""
        The merchant's original MCC code is {original_mcc} - {mcc_description}.
        Evaluate if this MCC is appropriate or if another MCC would be more suitable.
        """
        
        # Add AI description if available
        if ai_original_description:
            prompt += f"""
        Additional business description: {ai_original_description}
        """
        
        prompt += f"""
        Please assess the most appropriate MCC based on the following guidelines:
        - If the merchant name is similar or identical to the legal representative's name without specific industry indication, suggest a general service MCC.
        - If merchant name explicitly indicates a specific business category, suggest the most common MCC from the provided examples below.
        - If unclear or ambiguous, suggest a general MCC that would be most common.
        - IMPORTANT: Only use generic MCCs like 7299 or 5999 as a last resort if no more specific category applies.
        - Try to be as specific as possible based on the merchant name and any additional information provided.

        Here is a reference of commonly used MCC codes:

        {mcc_examples}

        **Always prioritize common MCCs for your suggestions.**
        **Consider that these businesses are mostly solo entrepreneurs, so avoid suggesting MCCs related to big companies or large enterprises.**

        **Respond strictly in the following format:**
        1. Analysis: [Brief analysis, highlighting alignment or mismatch]
        2. Suggested MCC: [Only MCC number]
        3. Suggested Description: [MCC description]
        """
        
        return prompt
    
    def _parse_openai_response(self, response_text: str) -> Tuple[str, str, str]:
        """
        Parse the OpenAI response to extract the suggested MCC and description.
        
        Args:
            response_text: The raw response from OpenAI
            
        Returns:
            Tuple of (analysis, suggested_mcc, suggested_description)
        """
        analysis = ""
        suggested_mcc = ""
        suggested_description = ""
        
        try:
            # Parse the response to extract analysis, suggested MCC, and description
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("1. Analysis:"):
                    analysis = line.replace("1. Analysis:", "").strip()
                elif line.startswith("2. Suggested MCC:"):
                    suggested_mcc = line.replace("2. Suggested MCC:", "").strip()
                elif line.startswith("3. Suggested Description:"):
                    suggested_description = line.replace("3. Suggested Description:", "").strip()
            
            # Handle case where the response says "Same"
            if suggested_mcc.lower() == "same":
                suggested_mcc = "7299"  # Default to Other Services
                suggested_description = "Other Services"
                
            # Clean up the MCC code (remove any non-digit characters)
            suggested_mcc = re.sub(r'[^0-9]', '', suggested_mcc)
            
            # If no valid MCC was extracted, use a default
            if not suggested_mcc:
                suggested_mcc = "7299"
                suggested_description = "Other Services"
                analysis = "Unable to determine specific MCC from business name."
        
        except Exception as e:
            self.logger.error(f"Error parsing OpenAI response: {str(e)}")
            suggested_mcc = "7299"  # Default to Other Services
            suggested_description = "Other Services"
            analysis = f"Error analyzing merchant: {str(e)}"
        
        return analysis, suggested_mcc, suggested_description
    
    def classify_with_openai(self, merchant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a merchant using OpenAI's GPT-4.
        
        Args:
            merchant_data: The full merchant record
            
        Returns:
            Dictionary with classification results
        """
        merchant_name = merchant_data.get('merchant_name', '')
        legal_name = merchant_data.get('legal_name', '')
        
        if not OPENAI_AVAILABLE or not self.client:
            self.logger.warning("OpenAI client not available. Using fallback classification.")
            return self._fallback_classify(merchant_data)
        
        try:
            # Create the prompt
            prompt = self._create_prompt(merchant_data)
            
            # Make the API call
            self.logger.info(f"Sending request to OpenAI for merchant: {merchant_name}")
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in merchant category codes (MCC). Provide your analysis and suggestions in the exact format requested. Avoid using generic categories like 7299 unless absolutely necessary."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            self.logger.debug(f"OpenAI response: {response_text}")
            
            # Parse the response
            analysis, suggested_mcc, suggested_description = self._parse_openai_response(response_text)
            
            # Log the classification result
            self.logger.info(f"Classified '{merchant_name}' as MCC {suggested_mcc} - {suggested_description}")
            
            # Determine confidence based on analysis
            if "unclear" in analysis.lower() or "ambiguous" in analysis.lower():
                confidence = 0.7
            elif "strongly" in analysis.lower() or "clearly" in analysis.lower():
                confidence = 0.95
            else:
                confidence = 0.85
            
            # Generate alternative MCCs
            alternative_mccs = self._generate_alternatives(suggested_mcc, merchant_name)
            
            # Determine industry classification
            industry = self._determine_industry(suggested_mcc)
            
            return {
                'mcc_code': suggested_mcc,
                'mcc_description': suggested_description,
                'confidence': confidence,
                'alternative_mccs': alternative_mccs,
                'analysis': analysis,
                'industry_classification': industry,
                'reasoning': f"Classification based on merchant name analysis."
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying with OpenAI: {str(e)}", exc_info=True)
            self.logger.error(f"Request data: merchant_name='{merchant_name}', legal_name='{legal_name}'")
            return self._fallback_classify(merchant_data)
    
    def _generate_alternatives(self, primary_mcc: str, merchant_name: str) -> List[Dict[str, Any]]:
        """
        Generate alternative MCC suggestions.
        
        Args:
            primary_mcc: The primary suggested MCC
            merchant_name: The merchant name
            
        Returns:
            List of alternative MCC dictionaries
        """
        # Convert merchant name to lowercase for matching
        merchant_lower = merchant_name.lower()
        
        # Create a list of candidate MCCs with their scores
        candidates = []
        
        # Strategy 1: Add candidates based on keyword matches in merchant name
        for mcc, description in self.mcc_data.items():
            # Skip the primary MCC
            if mcc == primary_mcc:
                continue
                
            # Simple keyword matching based on the MCC description
            desc_lower = description.lower()
            words = desc_lower.split()
            
            # Calculate a score based on word matches
            score = sum(1 for word in words if len(word) > 3 and word in merchant_lower)
            if score > 0:
                candidates.append((mcc, description, score))
        
        # Strategy 2: Add candidates based on industry similarity
        # Group MCCs by first two digits (industry category)
        if len(primary_mcc) >= 2:
            primary_category = primary_mcc[:2]
            for mcc, description in self.mcc_data.items():
                if mcc != primary_mcc and mcc.startswith(primary_category):
                    # Add a moderate score for being in the same category
                    candidates.append((mcc, description, 0.5))
        
        # If we don't have enough candidates, add some generic ones
        if len(candidates) < 2:
            if primary_mcc != "7299":
                candidates.append(("7299", "Other Services", 0))
            if primary_mcc != "5399":
                candidates.append(("5399", "Other Retail", 0))
            if primary_mcc != "7399":
                candidates.append(("7399", "Other B2B Services", 0))
        
        # Remove duplicates (keeping highest score)
        unique_candidates = {}
        for mcc, description, score in candidates:
            if mcc not in unique_candidates or score > unique_candidates[mcc][1]:
                unique_candidates[mcc] = (description, score)
        
        # Convert back to list and sort by score (highest first)
        candidates = [(mcc, desc, score) for mcc, (desc, score) in unique_candidates.items()]
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Take the top 2 candidates
        result = []
        for mcc, description, score in candidates[:2]:
            # Calculate confidence based on score
            base_confidence = 0.3
            score_factor = min(0.4, score * 0.1)  # Cap at 0.4
            confidence = round(base_confidence + score_factor, 2)
            
            result.append({
                'mcc_code': mcc,
                'mcc_description': description,
                'confidence': confidence,
                'explanation': f"Alternative classification based on merchant name similarity"
            })
        
        return result
    
    def _fallback_classify(self, merchant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback classification method when OpenAI is not available.
        
        Args:
            merchant_data: The full merchant record
            
        Returns:
            Dictionary with classification results
        """
        merchant_name = merchant_data.get('merchant_name', '')
        legal_name = merchant_data.get('legal_name', '')
        
        self.logger.info(f"Using fallback classification for: {merchant_name}")
        
        # Convert to lowercase for matching
        merchant_lower = merchant_name.lower()
        legal_name_lower = legal_name.lower() if legal_name else ''
        
        # Combine for matching
        combined_text = f"{merchant_lower} {legal_name_lower}"
        
        # Check for common keywords in the merchant name
        mcc_scores = {}
        for mcc, description in self.mcc_data.items():
            desc_lower = description.lower()
            keywords = desc_lower.split()
            
            # Calculate a score based on keyword matches
            score = sum(1 for keyword in keywords if len(keyword) > 3 and keyword in combined_text)
            if score > 0:
                mcc_scores[mcc] = score
        
        # If no matches, use a default MCC
        if not mcc_scores:
            return {
                'mcc_code': "7299",
                'mcc_description': "Other Services",
                'confidence': 0.50,
                'alternative_mccs': [
                    {'mcc_code': '5399', 'mcc_description': 'Other Retail', 'confidence': 0.2},
                    {'mcc_code': '7399', 'mcc_description': 'Other B2B Services', 'confidence': 0.15}
                ],
                'analysis': "No clear business category identified from name. Using generic service category."
            }
        
        # Get the top match
        top_mcc = max(mcc_scores.items(), key=lambda x: x[1])[0]
        description = self.mcc_data[top_mcc]
        
        # Calculate confidence based on the top score
        confidence = min(0.85, 0.5 + (mcc_scores[top_mcc] * 0.1))
        
        # Generate alternatives (excluding the top match)
        alternatives = []
        for mcc, score in sorted(mcc_scores.items(), key=lambda x: x[1], reverse=True):
            if mcc != top_mcc and len(alternatives) < 2:
                alt_confidence = min(0.6, 0.3 + (score * 0.1))
                alternatives.append({
                    'mcc_code': mcc,
                    'mcc_description': self.mcc_data[mcc],
                    'confidence': alt_confidence
                })
        
        # If we don't have enough alternatives, add a generic one
        if len(alternatives) < 2:
            if top_mcc != "7299":
                alternatives.append({
                    'mcc_code': '7299',
                    'mcc_description': 'Other Services',
                    'confidence': 0.2
                })
            else:
                alternatives.append({
                    'mcc_code': '5399',
                    'mcc_description': 'Other Retail',
                    'confidence': 0.2
                })
        
        return {
            'mcc_code': top_mcc,
            'mcc_description': description,
            'confidence': confidence,
            'alternative_mccs': alternatives,
            'analysis': f"Based on business name, classified as {description}."
        }
    
    def classify(self, merchant_name, legal_name=None, **merchant_data):
        """
        Classify a merchant using Rafa's algorithm with OpenAI.
        
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
        
        # If names are empty strings, return a default classification
        if not merchant_name.strip():
            return {
                'mcc_code': "7299",
                'mcc_description': "Other Services",
                'confidence': 0.5,
                'alternative_mccs': [
                    {'mcc_code': '5399', 'mcc_description': 'Other Retail', 'confidence': 0.3},
                    {'mcc_code': '7399', 'mcc_description': 'Other B2B Services', 'confidence': 0.2}
                ],
                'industry_classification': 'Services',
                'analysis': 'No merchant name provided.',
                'reasoning': 'Default classification due to missing merchant name.'
            }
        
        # Use OpenAI classification
        result = self.classify_with_openai(full_merchant_data)
        
        # Return the complete classification result
        return result
    
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
            
        if mcc.startswith("5"):
            return "Retail/Merchants"
        elif mcc.startswith("7"):
            return "Services"
        elif mcc.startswith("8"):
            return "Professional Services/Healthcare"
        elif mcc.startswith("4"):
            return "Transportation/Utilities"
        elif mcc.startswith("6"):
            return "Financial Services"
        elif mcc.startswith("9"):
            return "Government Services"
        else:
            return "Other Business Categories" 