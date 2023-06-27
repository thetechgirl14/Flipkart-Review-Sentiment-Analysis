#!/usr/bin/env python
# coding: utf-8


import unittest
import pandas as pd
import nltk
from main import load_the_dataset, pre_processing, distribution_of_ratings, add_new_columns, sentiment_scores

nltk.download('vader_lexicon')

class TestFlipkartSentimentAnalysis(unittest.TestCase):
    def test_load_the_dataset(self):
        data = load_the_dataset()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)

    def test_pre_processing(self):
        data = pre_processing()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.isnull().values.any())

    def test_distribution_of_ratings(self):
        ratings = distribution_of_ratings()
        self.assertIsNotNone(ratings)
        self.assertIsInstance(ratings, pd.Series)
        self.assertGreater(len(ratings), 0)

    def test_add_new_columns(self):
        data = add_new_columns()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('Positive', data.columns)
        self.assertIn('Negative', data.columns)
        self.assertIn('Neutral', data.columns)

    def test_sentiment_scores(self):
        x, y, z = sentiment_scores()
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertGreaterEqual(z, 0)


if __name__ == '__main__':
    
    unittest.main()

