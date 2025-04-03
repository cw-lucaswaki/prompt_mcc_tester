import csv
import logging
import os
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger("mcc_classifier.data_handler")

class DataHandler:
    """
    Utility class for handling data operations such as reading and writing CSV files.
    """
    
    @staticmethod
    def read_csv(file_path: str) -> List[Dict[str, Any]]:
        """
        Read a CSV file and return its contents as a list of dictionaries.
        
        Args:
            file_path (str): The path to the CSV file.
            
        Returns:
            list: A list of dictionaries, where each dictionary represents a row in the CSV.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: For other errors encountered while reading the file.
        """
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            logger.info(f"Reading data from {file_path}")
            
            # Use pandas to read the CSV file
            df = pd.read_csv(file_path)
            data = df.to_dict(orient="records")
            
            logger.info(f"Successfully read {len(data)} rows from {file_path}")
            return data
        
        except Exception as e:
            error_msg = f"Error reading CSV file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    @staticmethod
    def write_csv(file_path: str, data: List[Dict[str, Any]], fieldnames: List[str] = None) -> None:
        """
        Write data to a CSV file.
        
        Args:
            file_path (str): The path to the output CSV file.
            data (list): A list of dictionaries to write to the CSV file.
            fieldnames (list, optional): The field names to use for the CSV header.
                If not provided, the keys of the first dictionary in the data list will be used.
                
        Raises:
            ValueError: If data is empty or fieldnames are missing when data is empty.
            Exception: For other errors encountered while writing the file.
        """
        # Check for empty data without fieldnames
        if not data and not fieldnames:
            error_msg = "Cannot write CSV: data is empty and no fieldnames provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            logger.info(f"Writing {len(data)} rows to {file_path}")
            
            # Use pandas to write the CSV file
            df = pd.DataFrame(data)
            
            # Ensure columns are in the specified order if fieldnames is provided
            if fieldnames:
                # Add any missing columns to the dataframe
                for field in fieldnames:
                    if field not in df.columns:
                        df[field] = ""
                
                # Reorder columns according to fieldnames
                df = df[fieldnames]
            
            df.to_csv(file_path, index=False)
            
            logger.info(f"Successfully wrote data to {file_path}")
        
        except Exception as e:
            error_msg = f"Error writing CSV file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) 