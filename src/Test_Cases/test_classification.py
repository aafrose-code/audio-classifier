import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code_Logic'))
from src.Code_Logic.classification import CustomRuleBasedClassifier, GenderClassifier
#test the custom rule based classifier
class TestCustomRuleBasedClassifier(unittest.TestCase):
    #setup the classifier   
    def setUp(self):
        self.classifier = CustomRuleBasedClassifier()
    #test the default initialization
    def test_init_default(self):
        #test the default initialization
        self.assertEqual(self.classifier.pitch_threshold, 170.0)
        #test the default initialization
        self.assertAlmostEqual(self.classifier.mean_pitch_weight, 0.4 / 0.55, places=5)
        #test the default initialization
        self.assertAlmostEqual(self.classifier.pitch_std_weight, 0.15 / 0.55, places=5)

    def test_init_custom(self):
        #test the custom initialization
        classifier = CustomRuleBasedClassifier(
            pitch_threshold=180.0,
            mean_pitch_weight=0.8,
            pitch_std_weight=0.2
        )
        #test the custom initialization
        self.assertEqual(classifier.pitch_threshold, 180.0)
        #test the custom initialization
        self.assertEqual(classifier.mean_pitch_weight, 0.8)
        #test the custom initialization
        self.assertEqual(classifier.pitch_std_weight, 0.2)

    def test_init_weight_normalization(self):
        #test the init weight normalization
        classifier = CustomRuleBasedClassifier(
            mean_pitch_weight=0.5,
            pitch_std_weight=0.5
        )
        #test the init weight normalization
        self.assertAlmostEqual(classifier.mean_pitch_weight + classifier.pitch_std_weight, 1.0)

    def test_compute_segment_score_above_threshold(self):
        #test the compute segment score above threshold
        score = self.classifier.compute_segment_score(200.0, 15.0)
        #test the compute segment score above threshold
        self.assertGreater(score, 0)

    def test_compute_segment_score_below_threshold(self):
        #test the compute segment score below threshold
        score = self.classifier.compute_segment_score(120.0, 10.0)
        #test the compute segment score below threshold
        self.assertLess(score, 0)

    def test_compute_segment_score_at_threshold(self):
        #test the compute segment score at threshold
        score = self.classifier.compute_segment_score(170.0, 0.0)
        #test the compute segment score at threshold
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_classify_missing_segments(self):
        #test the classify missing segments
        with self.assertRaises(ValueError) as context:
            self.classifier.classify({'other': 'data'})
        #test the classify missing segments
        self.assertIn("segments", str(context.exception))
    #test the classify empty segments
    def test_classify_empty_segments(self):
        with self.assertRaises(ValueError) as context:
            self.classifier.classify({'segments': []})
        #test the classify empty segments       
        self.assertIn("No segments", str(context.exception))

    def test_classify_female_majority(self):
        #test the classify female majority
        features = {
            'segments': [#all the segments are female (high pitch and low std)
                {'mean_pitch': 200.0, 'pitch_std': 15.0},
                {'mean_pitch': 180.0, 'pitch_std': 12.0},
                {'mean_pitch': 190.0, 'pitch_std': 14.0}
            ]
        }
        result = self.classifier.classify(features)
        #test the classify female majority
        self.assertEqual(result['prediction'], 'female')
        #test the classify female majority
        self.assertEqual(result['female_votes'], 3)
        #test the classify female majority
        self.assertEqual(result['male_votes'], 0)

    def test_classify_male_majority(self):
        #test the classify male majority
        features = {
            'segments': [
                {'mean_pitch': 120.0, 'pitch_std': 8.0},
                {'mean_pitch': 130.0, 'pitch_std': 9.0},
                {'mean_pitch': 125.0, 'pitch_std': 7.0}
            ]
        }
        result = self.classifier.classify(features)
        #test the classify male majority

        self.assertEqual(result['prediction'], 'male')
        #test the classify male majority
        self.assertEqual(result['male_votes'], 3)
        #test the classify male majority
        self.assertEqual(result['female_votes'], 0)

    def test_classify_confidence_calculation(self):
        #test the classify confidence calculation
        features = {
            'segments': [
                {'mean_pitch': 200.0, 'pitch_std': 15.0},
                {'mean_pitch': 180.0, 'pitch_std': 12.0},
                {'mean_pitch': 190.0, 'pitch_std': 14.0},
                {'mean_pitch': 120.0, 'pitch_std': 8.0}
            ]
        }
        result = self.classifier.classify(features)
        #test the classify confidence calculation
        self.assertEqual(result['confidence'], 0.75)
        #test the classify confidence calculation
        self.assertEqual(result['num_segments'], 4)

    def test_classify_batch(self):
        #test the classify batch
        features_list = [
            {'segments': [{'mean_pitch': 200.0, 'pitch_std': 15.0}]},
            {'segments': [{'mean_pitch': 120.0, 'pitch_std': 8.0}]}
        ]
        #test the classify batch
        results = self.classifier.classify_batch(features_list)
        #test the classify batch
        self.assertEqual(len(results), 2)
        #test the classify batch
        self.assertEqual(results[0]['prediction'], 'female')
        #test the classify batch    
        self.assertEqual(results[1]['prediction'], 'male')
    #test the gender classifier
class TestGenderClassifier(unittest.TestCase):
    #setup the classifier
    def setUp(self):
        self.classifier = GenderClassifier()
    #test the default initialization
    def test_init_default(self):
        #test the default initialization
        self.assertEqual(self.classifier.classifier.pitch_threshold, 170.0)

    def test_init_custom(self):
        #test the custom initialization
        classifier = GenderClassifier(
            pitch_threshold=180.0,
            mean_pitch_weight=0.8,
            pitch_std_weight=0.2
        )
        #test the custom initialization
        self.assertEqual(classifier.classifier.pitch_threshold, 180.0)

    def test_classify(self):
        #test the classify
        features = {
            'segments': [
                {'mean_pitch': 200.0, 'pitch_std': 15.0}
            ]
        }
        result = self.classifier.classify(features)
        #test the classify
        self.assertEqual(result['final_prediction'], result['prediction'])
        #test the classify 
        self.assertEqual(result['method'], 'custom_rule_based')