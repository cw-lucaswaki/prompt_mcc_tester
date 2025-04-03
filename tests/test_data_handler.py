import os
import pytest
import pandas as pd
from mcc_classifier.utils.data_handler import DataHandler

@pytest.fixture
def sample_data():
    """Return sample data for testing."""
    return [
        {
            "Merchant Name": "Test Merchant 1",
            "Legal Name": "Test Legal 1",
            "Actual MCC code": "5411",
            "MCC Description": "Grocery Stores"
        },
        {
            "Merchant Name": "Test Merchant 2",
            "Legal Name": "Test Legal 2",
            "Actual MCC code": "5814",
            "MCC Description": "Fast Food Restaurants"
        }
    ]

@pytest.fixture
def temp_csv_path(tmp_path):
    """Create a temporary CSV file path."""
    return os.path.join(tmp_path, "test_data.csv")

def test_write_and_read_csv(sample_data, temp_csv_path):
    """Test writing to and reading from a CSV file."""
    # Write sample data to CSV
    DataHandler.write_csv(temp_csv_path, sample_data)
    
    # Check that the file exists
    assert os.path.exists(temp_csv_path)
    
    # Read data from CSV
    read_data = DataHandler.read_csv(temp_csv_path)
    
    # Check that the data is correct
    assert len(read_data) == len(sample_data)
    
    # Check each row
    for i, row in enumerate(sample_data):
        for key, value in row.items():
            assert str(read_data[i][key]) == str(value)

def test_write_csv_with_fieldnames(sample_data, temp_csv_path):
    """Test writing to a CSV file with specified fieldnames."""
    # Define fieldnames in a specific order
    fieldnames = ["MCC Description", "Actual MCC code", "Legal Name", "Merchant Name"]
    
    # Write sample data to CSV with specified fieldnames
    DataHandler.write_csv(temp_csv_path, sample_data, fieldnames)
    
    # Check that the file exists
    assert os.path.exists(temp_csv_path)
    
    # Read the CSV as a pandas DataFrame to check column order
    df = pd.read_csv(temp_csv_path)
    
    # Check that the columns are in the specified order
    assert list(df.columns) == fieldnames

def test_read_nonexistent_file():
    """Test reading a non-existent file."""
    with pytest.raises(FileNotFoundError):
        DataHandler.read_csv("nonexistent_file.csv")

def test_write_empty_data(temp_csv_path):
    """Test writing empty data to a CSV file."""
    # Write empty data with specified fieldnames
    fieldnames = ["Column1", "Column2"]
    DataHandler.write_csv(temp_csv_path, [], fieldnames)
    
    # Check that the file exists
    assert os.path.exists(temp_csv_path)
    
    # Read the CSV as a pandas DataFrame
    df = pd.read_csv(temp_csv_path)
    
    # Check that the DataFrame is empty but has the specified columns
    assert df.empty
    assert list(df.columns) == fieldnames

def test_write_empty_data_no_fieldnames(temp_csv_path):
    """Test writing empty data to a CSV file without fieldnames."""
    with pytest.raises(ValueError):
        DataHandler.write_csv(temp_csv_path, []) 