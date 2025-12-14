"""
    CS181DV Final Project: Interactive Data Visualization System

    Author: AIKO KATO

    Date: 05/07/2025
    
"""

import unittest
import pandas as pd
import sys
import os

# Add path to src so we can import data_processing.process_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now we can import from the package
from data_processing import process_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data_processing', 'Global_AI_Content_Impact_Dataset.csv')
        self.df = pd.read_csv(path)

    def test_clean_data_removes_nulls(self):
        cleaned = process_data.clean_data(self.df.copy())
        self.assertFalse(cleaned.isnull().any().any(), "Cleaned data should have no null values in required fields")

    def test_summary_country_has_expected_columns(self):
        cleaned = process_data.clean_data(self.df.copy())
        summary_country, _, _ = process_data.compute_summary_stats(cleaned)
        self.assertIn('Country', summary_country.columns)
        self.assertIn('AI Adoption Rate (%)', summary_country.columns)

    def test_summary_industry_grouping(self):
        cleaned = process_data.clean_data(self.df.copy())
        _, summary_industry, _ = process_data.compute_summary_stats(cleaned)
        self.assertGreater(len(summary_industry), 1, "Should summarize over multiple industries")

    def test_summary_tool_grouping(self):
        cleaned = process_data.clean_data(self.df.copy())
        _, _, summary_tool = process_data.compute_summary_stats(cleaned)
        self.assertGreater(len(summary_tool), 1, "Should summarize over multiple AI tools")

if __name__ == '__main__':
    unittest.main()
