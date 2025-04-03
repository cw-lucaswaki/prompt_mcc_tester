#!/usr/bin/env python3
"""
MCC Classification Evaluation Tool

This script evaluates the performance of different AI agents in classifying
merchants with their correct Merchant Category Codes (MCCs).
"""

import sys
from mcc_classifier.main import main

if __name__ == "__main__":
    sys.exit(main()) 