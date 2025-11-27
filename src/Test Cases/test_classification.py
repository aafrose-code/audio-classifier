import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code Logic'))
from classification import CustomRuleBasedClassifier, GenderClassifier


class TestCustomRuleBasedClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = CustomRuleBasedClassifier()
    
    def test_init_default(self):
        self.assertEqual(self.classifier.pitch_threshold, 165.0)
        self.assertEqual(self.classifier.mean_pitch_weight, 0.7)
        self.assertEqual(self.classifier.pitch_std_weight, 0.3)
    
    def test_init_custom(self):
        classifier = CustomRuleBasedClassifier(
            pitch_threshold=180.0,
            mean_pitch_weight=0.8,
            pitch_std_weight=0.2
        )
        self.assertEqual(classifier.pitch_threshold, 180.0)
        self.assertEqual(classifier.mean_pitch_weight, 0.8)
        self.assertEqual(classifier.pitch_std_weight, 0.2)
    
    def test_init_weight_normalization(self):
        classifier = CustomRuleBasedClassifier(
            mean_pitch_weight=0.5,
            pitch_std_weight=0.5
        )
        self.assertAlmostEqual(classifier.mean_pitch_weight + classifier.pitch_std_weight, 1.0)
    
    def test_init_zero_weights(self):
        classifier = CustomRuleBasedClassifier(
            mean_pitch_weight=0.0,
            pitch_std_weight=0.0
        )
        self.assertEqual(classifier.mean_pitch_weight, 0.0)
        self.assertEqual(classifier.pitch_std_weight, 0.0)
    
    def test_compute_segment_score_above_threshold(self):
        score = self.classifier.compute_segment_score(200.0, 15.0)
        self.assertGreater(score, 0)
    
    def test_compute_segment_score_below_threshold(self):
        score = self.classifier.compute_segment_score(120.0, 10.0)
        self.assertLess(score, 0)
    
    def test_compute_segment_score_at_threshold(self):
        score = self.classifier.compute_segment_score(165.0, 0.0)
        self.assertAlmostEqual(score, 0.0, places=5)
    
    def test_classify_segment_female(self):
        result = self.classifier.classify_segment(200.0, 20.0)
        self.assertEqual(result, 'female')
    
    def test_classify_segment_male(self):
        result = self.classifier.classify_segment(120.0, 10.0)
        self.assertEqual(result, 'male')
    
    def test_classify_segment_tie(self):
        result = self.classifier.classify_segment(165.0, 0.0)
        self.assertEqual(result, 'female')
    
    def test_classify_missing_segments(self):
        with self.assertRaises(ValueError) as context:
            self.classifier.classify({'other': 'data'})
        self.assertIn("segments", str(context.exception))
    
    def test_classify_empty_segments(self):
        with self.assertRaises(ValueError) as context:
            self.classifier.classify({'segments': []})
        self.assertIn("No segments", str(context.exception))
    
    def test_classify_female_majority(self):
        features = {
            'segments': [
                {'mean_pitch': 200.0, 'pitch_std': 15.0},
                {'mean_pitch': 180.0, 'pitch_std': 12.0},
                {'mean_pitch': 190.0, 'pitch_std': 14.0}
            ]
        }
        result = self.classifier.classify(features)
        self.assertEqual(result['prediction'], 'female')
        self.assertEqual(result['female_votes'], 3)
        self.assertEqual(result['male_votes'], 0)
    
    def test_classify_male_majority(self):
        features = {
            'segments': [
                {'mean_pitch': 120.0, 'pitch_std': 8.0},
                {'mean_pitch': 130.0, 'pitch_std': 9.0},
                {'mean_pitch': 125.0, 'pitch_std': 7.0}
            ]
        }
        result = self.classifier.classify(features)
        self.assertEqual(result['prediction'], 'male')
        self.assertEqual(result['male_votes'], 3)
        self.assertEqual(result['female_votes'], 0)
    
    def test_classify_tie_female_avg_score(self):
        features = {
            'segments': [
                {'mean_pitch': 200.0, 'pitch_std': 15.0},
                {'mean_pitch': 120.0, 'pitch_std': 8.0}
            ]
        }
        result = self.classifier.classify(features)
        self.assertIn(result['prediction'], ['male', 'female'])
        self.assertEqual(result['male_votes'], 1)
        self.assertEqual(result['female_votes'], 1)
    
    def test_classify_confidence_calculation(self):
        features = {
            'segments': [
                {'mean_pitch': 200.0, 'pitch_std': 15.0},
                {'mean_pitch': 180.0, 'pitch_std': 12.0},
                {'mean_pitch': 190.0, 'pitch_std': 14.0},
                {'mean_pitch': 120.0, 'pitch_std': 8.0}
            ]
        }
        result = self.classifier.classify(features)
        self.assertEqual(result['confidence'], 0.75)
        self.assertEqual(result['num_segments'], 4)
    
    def test_classify_batch(self):
        features_list = [
            {'segments': [{'mean_pitch': 200.0, 'pitch_std': 15.0}]},
            {'segments': [{'mean_pitch': 120.0, 'pitch_std': 8.0}]}
        ]
        results = self.classifier.classify_batch(features_list)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['prediction'], 'female')
        self.assertEqual(results[1]['prediction'], 'male')


class TestGenderClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = GenderClassifier()
    
    def test_init_default(self):
        self.assertEqual(self.classifier.classifier.pitch_threshold, 165.0)
    
    def test_init_custom(self):
        classifier = GenderClassifier(
            pitch_threshold=180.0,
            mean_pitch_weight=0.8,
            pitch_std_weight=0.2
        )
        self.assertEqual(classifier.classifier.pitch_threshold, 180.0)
    
    def test_classify(self):
        features = {
            'segments': [
                {'mean_pitch': 200.0, 'pitch_std': 15.0}
            ]
        }
        result = self.classifier.classify(features)
        self.assertEqual(result['final_prediction'], result['prediction'])
        self.assertEqual(result['method'], 'custom_rule_based')

