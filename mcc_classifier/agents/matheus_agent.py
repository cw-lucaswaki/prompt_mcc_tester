from mcc_classifier.agents.base_agent import MCCClassifierAgent
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import re
import pandas as pd
import csv
from pathlib import Path

try:
    from openai import OpenAI
    from pydantic import BaseModel, Field
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI or Pydantic package not installed. MatheusAgent will use fallback classification.")

# Define Pydantic models for OpenAI API responses
if OPENAI_AVAILABLE:
    class MCCClassification(BaseModel):
        suggested_mcc: str = Field(description="The suggested MCC code for the merchant")
        mcc_suggestion_description: str = Field(description="Description of the suggested MCC")
        analysis: List[str] = Field(description="List of reasoning steps explaining the analysis")
        confidence: float = Field(description="Confidence score (0-1) for the suggestion")
        suspicious_classification: bool = Field(description="Whether the original MCC seems intentionally misleading")

    class Tier1Result(BaseModel):
        suggested_mcc: str = Field(description="The suggested MCC code for the merchant")
        mcc_suggestion_description: str = Field(description="Description of the suggested MCC")
        analysis: List[str] = Field(description="List of reasoning steps explaining the analysis")
        confidence: float = Field(description="Confidence score (0-1) for the suggestion")
        requires_full_search: bool = Field(description="Whether a full MCC database search is recommended")
        is_non_descriptive: bool = Field(description="Whether the merchant name is essentially just the owner's name")
        may_be_high_risk: bool = Field(description="Whether the merchant name suggests a potentially high-risk business")
        suspicious_classification: bool = Field(description="Whether the original MCC seems intentionally misleading")

    class RiskTierResult(BaseModel):
        suggested_mcc: str = Field(description="The suggested MCC code for the merchant")
        mcc_suggestion_description: str = Field(description="Description of the suggested MCC")
        analysis: List[str] = Field(description="List of reasoning steps explaining the analysis")
        confidence: float = Field(description="Confidence score (0-1) for the suggestion")
        risk_level: str = Field(description="The risk level of the suggested MCC (prohibited, pre-approval, pre-review, restricted, approved)")
        requires_full_search: bool = Field(description="Whether a full MCC database search is recommended")
        suspicious_classification: bool = Field(description="Whether the original classification appears suspicious or intentionally misleading")

class MatheusAgent(MCCClassifierAgent):
    """
    Matheus's implementation of the MCC classifier agent.
    
    This agent uses OpenAI's models with a three-tier approach (limited MCC list, risk-based, and full MCC list)
    to classify merchants with appropriate MCCs.
    """
    
    def __init__(self):
        """Initialize the Matheus MCC classifier agent."""
        super().__init__("Matheus")
        
        # Initialize dictionaries
        self.mcc_data = {}
        self.mcc_risk_levels = {"default": "approved"}
        
        # Define catch-all MCCs that should only be used as a last resort
        self.catch_all_mccs = ["7299", "7399", "5399", "1799", "7999", "8999"]
        
        # Load MCC data from CSV file
        self._load_mcc_data()
        
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
    
    def _load_mcc_data(self):
        """
        Load MCC data from the mcc_list.csv file.
        """
        try:
            # Get the path to the CSV file (in the same directory as this file)
            csv_path = Path(__file__).parent / "mcc_list.csv"
            
            # Check if file exists
            if not csv_path.exists():
                self.logger.warning(f"MCC list file not found at {csv_path}. Using default MCC data.")
                self._initialize_default_mcc_data()
                return
            
            # Read the CSV file
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    mcc = row['mcc']
                    description = row['description']
                    classification = row.get('classification', 'approved').lower()
                    
                    # Populate mcc_data dictionary
                    self.mcc_data[mcc] = description
                    
                    # Populate mcc_risk_levels dictionary
                    self.mcc_risk_levels[mcc] = classification
            
            self.logger.info(f"Successfully loaded {len(self.mcc_data)} MCCs from {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading MCC data from CSV: {str(e)}")
            self._initialize_default_mcc_data()
    
    def _initialize_default_mcc_data(self):
        """Initialize with default MCC data in case CSV loading fails."""
        # Dictionary of common MCC codes and their descriptions (backup)
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
        
        # Classification of risk levels for MCCs (backup)
        self.mcc_risk_levels = {
            # High-risk MCCs (examples)
            "5933": "prohibited",  # Pawn Shops
            "7995": "prohibited",  # Gambling
            "5993": "prohibited",  # Cigar Stores and Stands
            "7273": "prohibited",  # Dating Services
            "7841": "restricted",  # Adult Video Stores
            "5816": "restricted",  # Digital Goods
            # Default for other MCCs
            "default": "approved"
        }
        
        self.logger.info("Initialized with default MCC data.")
    
    def _prepare_mcc_data(self):
        """
        Prepare MCC data for use in prompts.
        
        Returns:
            Tuple[str, str, str]: CSV strings for limited MCCs, restricted MCCs, and all MCCs
        """
        # Create CSV format strings for MCC data
        limited_mcc_csv = "mcc,description\n"
        restricted_mcc_csv = "mcc,description,classification\n"
        all_mcc_csv = "mcc,description,classification\n"
        
        # Add MCCs to each CSV format
        for mcc, description in self.mcc_data.items():
            risk_level = self.mcc_risk_levels.get(mcc, self.mcc_risk_levels["default"])
            
            # Add to all MCCs list
            all_mcc_csv += f"{mcc},{description},{risk_level}\n"
            
            # Add to limited MCCs list (exclude high-risk MCCs)
            if risk_level == "approved":
                limited_mcc_csv += f"{mcc},{description}\n"
            
            # Add to restricted MCCs list (only include higher-risk MCCs)
            if risk_level != "approved":
                restricted_mcc_csv += f"{mcc},{description},{risk_level}\n"
        
        return limited_mcc_csv, restricted_mcc_csv, all_mcc_csv 

    def _classify_tier1(self, merchant_name: str, legal_name: str, mcc: str, mcc_description: str, limited_mcc_list: str, **merchant_data) -> Dict[str, Any]:
        """
        First-tier classification using limited MCC list.
        
        Args:
            merchant_name: The merchant name
            legal_name: The legal name
            mcc: Current MCC code
            mcc_description: Current MCC description
            limited_mcc_list: CSV string with limited MCC data
            **merchant_data: Additional merchant data
            
        Returns:
            Dictionary with classification results
        """
        if not OPENAI_AVAILABLE or not self.client:
            self.logger.warning("OpenAI client not available for tier1 classification.")
            return self._fallback_classify(merchant_name, legal_name, **merchant_data)
        
        try:
            self.logger.info(f"Performing tier1 classification for: {merchant_name}")
            
            completion = self.client.beta.chat.completions.parse(
                model="o3-mini",  # Use the specified model
                messages=[
                    {"role": "system", "content": f"""
                        You are an expert in merchant category classification. Your task is to evaluate if a merchant's
                        auto-classification is accurate based on the provided information.

                        Analyze the owner's name, merchant name (business/store name), current MCC code, and MCC description.
                        Then determine:
                        1. If the merchant name is non-descriptive (SPECIFICALLY: if it's just the owner's name or a minor variation)
                        2. If the merchant name suggests a potentially high-risk business (adult content, gambling, etc.)
                        3. If the current MCC classification is appropriate
                        4. If not, suggest a more appropriate MCC from the provided limited list
                        5. Provide clear reasoning for your decision
                        6. Assign a confidence score (0-1) to your suggestion
                        7. Determine if a full search of all MCCs is needed
                        8. Determine if the original classification appears suspicious or intentionally misleading

                        IMPORTANT - SUSPICIOUS CLASSIFICATION:
                        Set suspicious_classification to TRUE if the original MCC has NO logical connection to the merchant name
                        and appears to be intentionally misleading. Examples:
                        * Merchant Name: "PIZZERIA NUEVA LEON", MCC: "Barber Shops" - This is suspicious
                        * Merchant Name: "GOLDEN NAIL SPA", MCC: "Family Clothing Stores" - This is suspicious
                        * Merchant Name: "AUTO PARTS CENTER", MCC: "Real Estate Agents" - This is suspicious

                        Only mark as suspicious if there is a clear disconnect that cannot be explained by ambiguity or
                        lack of available categories. A slight mismatch is not suspicious - it must be completely unrelated.

                        IMPORTANT: The following MCCs are CATCH-ALL CATEGORIES and should ONLY be used as a LAST RESORT:
                        {", ".join(self.catch_all_mccs)}

                        Only suggest these catch-all categories if NO other category in the limited list could possibly apply.
                        If you are considering a catch-all category, you should set requires_full_search to true so we can
                        try to find a more specific category in the complete MCC list.

                        IMPORTANT GUIDANCE ON HIGH-RISK BUSINESSES:
                        - If a merchant name suggests adult services, gambling, weapons, drugs, or other potentially
                          restricted activities, set may_be_high_risk to true
                        - Examples that should be flagged as potentially high risk:
                          * "Orgasmtron" (suggests adult services)
                          * "Wholesalewhores" (suggests adult services)
                          * "Lucky Slots" (suggests gambling)
                          * Names containing explicit sexual terminology

                        IMPORTANT GUIDANCE ON ANALYZING BUSINESS NAMES:
                        - Consider the FULL name and its likely meaning
                        - Be careful not to over-rely on partial word matches
                        - For names like "AUTOMATICWINBOYZ LLC" - don't assume it's clothing just because it has "BOYZ"
                          in it unless there are other clear indicators of clothing retail

                        IMPORTANT CLARIFICATION ON NON-DESCRIPTIVE NAMES:
                        - A merchant name is ONLY considered non-descriptive if it is essentially just the owner's name
                          or a very slight variation of it (e.g., "John Smith" → "John Smith LLC" or "J Smith Services")
                        - Examples where the name would be considered non-descriptive:
                          * Owner: "Robert Paulson", Merchant: "Robert Paulson LLC"
                          * Owner: "Robert Paulson", Merchant: "Bob Paulson"
                        - Examples where the name would NOT be considered non-descriptive:
                          * Owner: "Linda Carter", Merchant: "Labomba Lay"
                          * Owner: "John Smith", Merchant: "Sunrise Enterprises"

                        Here is the limited list of approved MCC codes for classification:

                        {limited_mcc_list}

                        Base your analysis on the context of the complete business name, not just partial matches.
                     """},
                    {"role": "user", "content": f"""
                        Please analyze this merchant:

                        Owner Name: {legal_name}
                        Merchant Name: {merchant_name}
                        Current MCC: {mcc}
                        MCC Description: {mcc_description}

                        First determine if the merchant name is essentially just the owner's name.
                        Then check if the name suggests a potentially high-risk business.
                        Then determine if the current classification makes sense or suggest a more appropriate one.
                        Also determine if the current classification appears suspicious (completely unrelated to the business).

                        IMPORTANT: Avoid using general "catch-all" categories unless absolutely necessary. If you're
                        considering a catch-all category, please indicate that a full search is needed instead.

                        Consider the merchant name in its entirety when making your assessment.
                     """},
                ],
                response_format=Tier1Result,
            )
            
            result = completion.choices[0].message.parsed
            
            # Convert from Pydantic model to dictionary
            return {
                'suggested_mcc': result.suggested_mcc,
                'mcc_suggestion_description': result.mcc_suggestion_description,
                'analysis': result.analysis,
                'confidence': result.confidence,
                'requires_full_search': result.requires_full_search,
                'is_non_descriptive': result.is_non_descriptive if hasattr(result, 'is_non_descriptive') else False,
                'may_be_high_risk': result.may_be_high_risk,
                'suspicious_classification': result.suspicious_classification
            }
            
        except Exception as e:
            self.logger.error(f"Error in tier1 classification: {str(e)}")
            return self._fallback_classify(merchant_name, legal_name) 

    def _classify_risk_tier(self, merchant_name: str, legal_name: str, mcc: str, mcc_description: str, 
                            restricted_mcc_list: str, may_be_high_risk: bool, **merchant_data) -> Dict[str, Any]:
        """
        Risk-tier classification using restricted MCCs list with risk classifications.
        
        Args:
            merchant_name: The merchant name
            legal_name: The legal name
            mcc: Current MCC code
            mcc_description: Current MCC description
            restricted_mcc_list: CSV string with restricted MCC data
            may_be_high_risk: Flag indicating if merchant may be high-risk
            **merchant_data: Additional merchant data
            
        Returns:
            Dictionary with classification results
        """
        if not OPENAI_AVAILABLE or not self.client:
            self.logger.warning("OpenAI client not available for risk tier classification.")
            return self._fallback_classify(merchant_name, legal_name, **merchant_data)
        
        try:
            self.logger.info(f"Performing risk tier classification for: {merchant_name}")
            
            completion = self.client.beta.chat.completions.parse(
                model="o3-mini",  # Use the specified model
                messages=[
                    {"role": "system", "content": f"""
                        You are an expert in merchant category classification with a focus on risk assessment.
                        Your task is to determine if a merchant should be classified in one of the risk-categorized MCCs.

                        The merchant has already been through an initial screening, and {
                        "has been flagged as potentially high-risk" if may_be_high_risk else "needs additional review"}.

                        Analyze the owner's name, merchant name, current MCC code, and MCC description, then determine:
                        1. The most appropriate MCC from the risk-categorized list
                        2. The risk level of that MCC (prohibited, pre-approval, pre-review, restricted, approved)
                        3. Provide clear reasoning for your decision
                        4. Assign a confidence score (0-1) to your suggestion
                        5. Determine if a full search of all MCCs is still needed
                        6. Determine if the original classification appears suspicious or intentionally misleading

                        IMPORTANT - SUSPICIOUS CLASSIFICATION:
                        Set suspicious_classification to TRUE if the original MCC has NO logical connection to the merchant name
                        and appears to be intentionally misleading. Examples:
                        * Merchant Name: "PIZZERIA NUEVA LEON", MCC: "Barber Shops" - This is suspicious
                        * Merchant Name: "GOLDEN NAIL SPA", MCC: "Family Clothing Stores" - This is suspicious
                        * Merchant Name: "AUTO PARTS CENTER", MCC: "Real Estate Agents" - This is suspicious

                        Only mark as suspicious if there is a clear disconnect that cannot be explained by ambiguity or
                        lack of available categories. A slight mismatch is not suspicious - it must be completely unrelated.

                        IMPORTANT: The following MCCs are CATCH-ALL CATEGORIES and should ONLY be used as a LAST RESORT:
                        {", ".join(self.catch_all_mccs)}

                        Only suggest these catch-all categories if NO other category could possibly apply.

                        RISK PRIORITIZATION:
                        - When choosing between multiple possible MCCs, ALWAYS prioritize by risk level:
                          1. First try to match to "prohibited" MCCs
                          2. Then "pre-approval" MCCs
                          3. Then "pre-review" MCCs
                          4. Then "restricted" MCCs
                          5. Only use "approved" MCCs if nothing else fits

                        - For adult-oriented businesses, massage parlors (if available) are often appropriate
                        - For gambling-related businesses, assign to gambling-related MCCs even if confidence is moderate

                        Here is the list of MCCs with risk classifications:

                        {restricted_mcc_list}

                        IMPORTANT: Always prioritize higher-risk classifications when in doubt.
                     """},
                    {"role": "user", "content": f"""
                        Please analyze this merchant for potential risk classification:

                        Owner Name: {legal_name}
                        Merchant Name: {merchant_name}
                        Current MCC: {mcc}
                        MCC Description: {mcc_description}

                        {
                        "The merchant name has been flagged as potentially high-risk. " if may_be_high_risk else ""
                        }Determine the most appropriate risk-categorized MCC, prioritizing higher-risk classifications.
                        If you find a reasonable match in a higher-risk category, prefer that over lower-risk categories.

                        Also determine if the current MCC classification appears suspicious (completely unrelated to the business).

                        IMPORTANT: Avoid using general "catch-all" categories unless absolutely necessary. If you're
                        considering a catch-all category, please indicate that a full search is needed instead.
                     """},
                ],
                response_format=RiskTierResult,
            )
            
            result = completion.choices[0].message.parsed
            
            # Convert from Pydantic model to dictionary
            return {
                'suggested_mcc': result.suggested_mcc,
                'mcc_suggestion_description': result.mcc_suggestion_description,
                'analysis': result.analysis,
                'confidence': result.confidence,
                'risk_level': result.risk_level,
                'requires_full_search': result.requires_full_search,
                'suspicious_classification': result.suspicious_classification
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk tier classification: {str(e)}")
            return self._fallback_classify(merchant_name, legal_name, **merchant_data)

    def _classify_tier3(self, merchant_name: str, legal_name: str, mcc: str, mcc_description: str,
                        all_mcc_list: str, suspicious_classification_flag: bool = False, **merchant_data) -> Dict[str, Any]:
        """
        Final-tier classification using full MCC list.
        
        Args:
            merchant_name: The merchant name
            legal_name: The legal name
            mcc: Current MCC code
            mcc_description: Current MCC description
            all_mcc_list: CSV string with all MCC data
            suspicious_classification_flag: Flag from previous classification steps
            **merchant_data: Additional merchant data fields
            
        Returns:
            Dictionary with classification results
        """
        if not OPENAI_AVAILABLE or not self.client:
            self.logger.warning("OpenAI client not available for tier3 classification.")
            return self._fallback_classify(merchant_name, legal_name, **merchant_data)
        
        try:
            self.logger.info(f"Performing tier3 classification for: {merchant_name}")
            
            completion = self.client.beta.chat.completions.parse(
                model="o3-mini",  # Use the specified model
                messages=[
                    {"role": "system", "content": f"""
                        You are an expert in merchant category classification. Your task is to find the most appropriate
                        MCC classification for a merchant from the complete list of MCC codes.

                        The merchant has already been evaluated against a limited list and risk-categorized list,
                        but requires a full analysis against all MCCs.

                        Analyze the owner's name, merchant name (business/store name), current MCC code, and MCC description.
                        Then determine:
                        1. The most appropriate MCC from the complete list
                        2. Provide clear reasoning for your decision
                        3. Assign a confidence score (0-1) to your suggestion
                        4. Determine if the original classification appears suspicious or intentionally misleading
                           (you've been given a preliminary assessment: {suspicious_classification_flag})

                        IMPORTANT - SUSPICIOUS CLASSIFICATION:
                        Set suspicious_classification to TRUE if the original MCC has NO logical connection to the merchant name
                        and appears to be intentionally misleading. Examples:
                        * Merchant Name: "PIZZERIA NUEVA LEON", MCC: "Barber Shops" - This is suspicious
                        * Merchant Name: "GOLDEN NAIL SPA", MCC: "Family Clothing Stores" - This is suspicious
                        * Merchant Name: "AUTO PARTS CENTER", MCC: "Real Estate Agents" - This is suspicious

                        Only mark as suspicious if there is a clear disconnect that cannot be explained by ambiguity or
                        lack of available categories. A slight mismatch is not suspicious - it must be completely unrelated.

                        IMPORTANT: The following MCCs are CATCH-ALL CATEGORIES and should ONLY be used as a LAST RESORT:
                        {", ".join(self.catch_all_mccs)}

                        Only suggest these catch-all categories if NO other category could possibly apply.
                        Look for the most specific category that matches the business type.

                        IMPORTANT GUIDANCE ON ANALYZING BUSINESS NAMES:
                        - Consider the FULL name and its likely intended meaning
                        - Make classifications based on the most likely business type given the complete name
                        - For ambiguous names, acknowledge this in your analysis

                        Here is the complete list of MCC codes:

                        {all_mcc_list}

                        Base your analysis on the context of the complete business name, not just partial matches.
                     """},
                    {"role": "user", "content": f"""
                        Please find the most appropriate MCC classification for this merchant from the complete list:

                        Owner Name: {legal_name}
                        Merchant Name: {merchant_name}
                        Current MCC: {mcc}
                        MCC Description: {mcc_description}

                        Consider the merchant name in its entirety when making your assessment.
                        Provide a thorough analysis with your confidence level.

                        Also determine if the current MCC classification appears suspicious (completely unrelated to the business).

                        IMPORTANT: Avoid using general "catch-all" categories unless absolutely necessary.
                        If the merchant name clearly indicates a specific business type (like "PRESTIGE PRO REAL ESTATE"
                        indicating a real estate business), find the most specific category for that business type.
                     """},
                ],
                response_format=MCCClassification,
            )
            
            result = completion.choices[0].message.parsed
            
            # Convert from Pydantic model to dictionary
            return {
                'suggested_mcc': result.suggested_mcc,
                'mcc_suggestion_description': result.mcc_suggestion_description,
                'analysis': result.analysis,
                'confidence': result.confidence,
                'suspicious_classification': result.suspicious_classification or suspicious_classification_flag
            }
            
        except Exception as e:
            self.logger.error(f"Error in tier3 classification: {str(e)}")
            return self._fallback_classify(merchant_name, legal_name, **merchant_data)
    
    def _fallback_classify(self, merchant_name: str, legal_name: str = None, **merchant_data) -> Dict[str, Any]:
        """
        Fallback classification method when OpenAI is not available.
        
        Args:
            merchant_name: The name of the merchant
            legal_name: The legal name of the merchant
            **merchant_data: Additional merchant data fields
            
        Returns:
            Dictionary with classification results
        """
        self.logger.info(f"Using fallback classification for: {merchant_name}")
        
        # Use original MCC if provided in merchant_data
        original_mcc = merchant_data.get('original_mcc_code', None)
        original_mcc_desc = merchant_data.get('mcc_description', None)
        
        # If we have original MCC data and it's valid, use it with reduced confidence
        if original_mcc and original_mcc in self.mcc_data:
            return {
                'suggested_mcc': original_mcc,
                'mcc_suggestion_description': original_mcc_desc or self.mcc_data[original_mcc],
                'confidence': 0.60,
                'analysis': ["Using provided MCC code as fallback classification."],
                'suspicious_classification': False,
                'is_non_descriptive': False,
                'may_be_high_risk': False,
                'requires_full_search': False
            }
            
        # Convert to lowercase for matching
        merchant_lower = merchant_name.lower()
        
        # Check for common keywords in the merchant name
        mcc_scores = {}
        for mcc, description in self.mcc_data.items():
            desc_lower = description.lower()
            keywords = desc_lower.split()
            
            # Calculate a score based on keyword matches
            score = sum(1 for keyword in keywords if len(keyword) > 3 and keyword in merchant_lower)
            if score > 0:
                mcc_scores[mcc] = score
        
        # If no matches, use a default MCC
        if not mcc_scores:
            return {
                'suggested_mcc': "5812",  # Restaurants - a common default for Matheus agent
                'mcc_suggestion_description': "Eating Places, Restaurants",
                'confidence': 0.50,
                'analysis': ["No clear business category identified from name. Using common restaurant category."],
                'suspicious_classification': False,
                'is_non_descriptive': True,
                'may_be_high_risk': False,
                'requires_full_search': True
            }
        
        # Get the top match
        top_mcc = max(mcc_scores.items(), key=lambda x: x[1])[0]
        description = self.mcc_data[top_mcc]
        
        # Calculate confidence based on the top score
        confidence = min(0.85, 0.5 + (mcc_scores[top_mcc] * 0.1))
        
        return {
            'suggested_mcc': top_mcc,
            'mcc_suggestion_description': description,
            'confidence': confidence,
            'analysis': [f"Based on keywords in the business name, classified as {description}."],
            'suspicious_classification': False,
            'is_non_descriptive': False,
            'may_be_high_risk': False,
            'requires_full_search': False
        }

    def classify(self, merchant_name, legal_name=None, **merchant_data):
        """
        Classify a merchant using Matheus's three-tier approach with risk prioritization.
        Complete merchant classification using three-tier approach with risk prioritization.
        
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
        
        # If names are empty strings, return a default classification
        if not merchant_name.strip():
            return {
                'mcc_code': "5812",  # Restaurant - Matheus's default
                'mcc_description': "Eating Places, Restaurants",
                'confidence': 0.5,
                'alternative_mccs': [
                    {'mcc_code': '5399', 'mcc_description': 'Other Retail', 'confidence': 0.3},
                    {'mcc_code': '7399', 'mcc_description': 'Other B2B Services', 'confidence': 0.2}
                ],
                'industry_classification': 'Restaurants',
                'analysis': ['No merchant name provided.'],
                'reasoning': 'Default classification due to missing merchant name.'
            }
        
        # Prepare MCC data for use in prompts
        limited_mcc_list, restricted_mcc_list, all_mcc_list = self._prepare_mcc_data()
        
        # Default MCC if none provided, or use the original_mcc_code if available
        mcc = merchant_data.get('original_mcc_code', "5812")
        mcc_description = merchant_data.get('mcc_description', "Eating Places, Restaurants")
        
        # First tier: Check against limited MCC list and assess name descriptiveness
        tier1_result = self._classify_tier1(merchant_name, legal_name, mcc, mcc_description, limited_mcc_list, **merchant_data)
        
        # If merchant name is non-descriptive, keep original MCC
        if tier1_result.get('is_non_descriptive', False):
            # We're keeping the original MCC, so prepare the final result
            final_result = {
                'mcc_code': mcc,
                'mcc_description': mcc_description,
                'confidence': 1.0,
                'analysis': ["NON-DESCRIPTIVE NAME MCC ACCEPTED"] + tier1_result.get('analysis', []),
                'suspicious_classification': False
            }
        else:
            # If suggested MCC is a catch-all and confidence isn't extremely high, force additional search
            if tier1_result.get('suggested_mcc') in self.catch_all_mccs and tier1_result.get('confidence', 0) < 0.98:
                tier1_result['requires_full_search'] = True
                if 'analysis' in tier1_result:
                    tier1_result['analysis'].append("Suggested MCC is a general catch-all category. Attempting to find a more specific classification.")
            
            # If first tier is confident AND not requiring full search AND not potentially high risk, use its result
            if (tier1_result.get('confidence', 0) >= 0.7 and 
                not tier1_result.get('requires_full_search', False) and 
                not tier1_result.get('may_be_high_risk', False)):
                
                final_result = {
                    'mcc_code': tier1_result.get('suggested_mcc'),
                    'mcc_description': tier1_result.get('mcc_suggestion_description'),
                    'confidence': tier1_result.get('confidence'),
                    'analysis': tier1_result.get('analysis', []),
                    'suspicious_classification': tier1_result.get('suspicious_classification', False)
                }
            else:
                # Risk tier: If potentially high risk or needs more review, check against restricted MCC list
                risk_tier_result = self._classify_risk_tier(
                    merchant_name, 
                    legal_name, 
                    mcc, 
                    mcc_description, 
                    restricted_mcc_list,
                    tier1_result.get('may_be_high_risk', False),
                    **merchant_data
                )
                
                # If the risk tier suggested a catch-all MCC, force a full search unless confidence is near-perfect
                if (risk_tier_result.get('suggested_mcc') in self.catch_all_mccs and 
                    risk_tier_result.get('confidence', 0) < 0.98):
                    risk_tier_result['requires_full_search'] = True
                    if 'analysis' in risk_tier_result:
                        risk_tier_result['analysis'].append("Suggested MCC is a general catch-all category. Attempting to find a more specific classification.")
                
                # If we found a good match in the risk tier with adequate confidence, use its result
                if not risk_tier_result.get('requires_full_search', False) and risk_tier_result.get('confidence', 0) >= 0.7:
                    analysis = risk_tier_result.get('analysis', [])
                    if risk_tier_result.get('risk_level', "approved") != "approved":
                        analysis.insert(0, f"ATTENTION: Business classified as {risk_tier_result.get('risk_level', '').upper()}")
                    
                    # Preserve suspicious classification flag from tier 1 if applicable
                    suspicious = (risk_tier_result.get('suspicious_classification', False) or 
                                 tier1_result.get('suspicious_classification', False))
                    
                    final_result = {
                        'mcc_code': risk_tier_result.get('suggested_mcc'),
                        'mcc_description': risk_tier_result.get('mcc_suggestion_description'),
                        'confidence': risk_tier_result.get('confidence'),
                        'analysis': analysis,
                        'suspicious_classification': suspicious
                    }
                else:
                    # Final tier: Complete MCC list search as last resort
                    suspicious_flag = (tier1_result.get('suspicious_classification', False) or 
                                      risk_tier_result.get('suspicious_classification', False))
                    
                    tier3_result = self._classify_tier3(
                        merchant_name, 
                        legal_name, 
                        mcc, 
                        mcc_description, 
                        all_mcc_list,
                        suspicious_flag,
                        **merchant_data
                    )
                    
                    final_result = {
                        'mcc_code': tier3_result.get('suggested_mcc'),
                        'mcc_description': tier3_result.get('mcc_suggestion_description'),
                        'confidence': tier3_result.get('confidence'),
                        'analysis': tier3_result.get('analysis', []),
                        'suspicious_classification': tier3_result.get('suspicious_classification', False)
                    }
        
        # Add alternative MCCs to the final result
        final_result['alternative_mccs'] = self.generate_alternative_mccs(final_result['mcc_code'], merchant_name, self.mcc_data)
        
        # Ensure mcc_code is used instead of suggested_mcc in the final output
        if 'suggested_mcc' in final_result:
            final_result['mcc_code'] = final_result.pop('suggested_mcc')
        
        # Ensure mcc_description is used instead of mcc_suggestion_description in the final output
        if 'mcc_suggestion_description' in final_result:
            final_result['mcc_description'] = final_result.pop('mcc_suggestion_description')
        
        # Add mcc_changed flag
        final_result['mcc_changed'] = str(final_result['mcc_code']) != str(mcc)
        
        # Add industry_classification based on the MCC code
        final_result['industry_classification'] = self._determine_industry(final_result['mcc_code'])
        
        # Add reasoning field derived from analysis
        if 'analysis' in final_result and final_result['analysis']:
            final_result['reasoning'] = " ".join(final_result['analysis'])
        else:
            final_result['reasoning'] = "Classification based on merchant name analysis"
        
        return final_result
    
    def generate_alternative_mccs(self, primary_mcc: str, merchant_name: str, mcc_data: dict) -> list:
        merchant_lower = merchant_name.lower()
        candidates = []
        
        # Add candidates based on keyword matches in merchant name
        for mcc, description in mcc_data.items():
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
        
        # If we don't have enough candidates, add some generic ones
        if len(candidates) < 2:
            # Default preferences
            if primary_mcc != "5812":
                candidates.append(("5812", "Eating Places, Restaurants", 0))
            if primary_mcc != "5399":
                candidates.append(("5399", "Other Retail", 0))
            if primary_mcc != "7399":
                candidates.append(("7399", "Other B2B Services", 0))
        
        # Sort candidates by score (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Take the top 2 candidates
        result = []
        for mcc, description, score in candidates[:2]:
            # Calculate confidence based on score
            base_confidence = 0.3
            score_factor = min(0.4, score * 0.1)  # Cap at 0.4
            confidence = base_confidence + score_factor
            
            result.append({
                'mcc_code': mcc,
                'mcc_description': description,
                'confidence': confidence
            })
        
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