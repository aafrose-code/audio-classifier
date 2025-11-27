import unittest
import sys
import os
import json
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code Logic'))
from main import GenderIdentificationSystem
from extract_voice_metadata import VoiceFeatureExtractor
from classification import GenderClassifier


class TestGenderIdentificationSystem(unittest.TestCase):
    
    def setUp(self):
        self.system = GenderIdentificationSystem()
    
    def test_init_default(self):
        self.assertIsNotNone(self.system.feature_extractor)
        self.assertIsNotNone(self.system.classifier)
        self.assertEqual(len(self.system.training_data), 0)
        self.assertEqual(len(self.system.test_data), 0)
    
    def test_init_custom(self):
        system = GenderIdentificationSystem(
            pitch_threshold=180.0,
            mean_pitch_weight=0.8,
            pitch_std_weight=0.2
        )
        self.assertIsNotNone(system.classifier)
    
    @patch.object(VoiceFeatureExtractor, 'extract_features_batch')
    def test_collect_features_no_labels(self, mock_extract):
        mock_extract.return_value = [
            {'segments': [], 'num_segments': 0, 'audio_path': 'test1.wav'},
            {'segments': [], 'num_segments': 0, 'audio_path': 'test2.wav'}
        ]
        
        result = self.system.collect_features(['test1.wav', 'test2.wav'])
        self.assertEqual(len(result), 2)
        self.assertNotIn('true_label', result[0])
    
    @patch.object(VoiceFeatureExtractor, 'extract_features_batch')
    def test_collect_features_with_labels(self, mock_extract):
        mock_extract.return_value = [
            {'segments': [], 'num_segments': 0, 'audio_path': 'test1.wav'},
            {'segments': [], 'num_segments': 0, 'audio_path': 'test2.wav'}
        ]
        
        result = self.system.collect_features(['test1.wav', 'test2.wav'], ['male', 'female'])
        self.assertEqual(result[0]['true_label'], 'male')
        self.assertEqual(result[1]['true_label'], 'female')
    
    @patch.object(VoiceFeatureExtractor, 'extract_features_batch')
    def test_collect_features_label_mismatch(self, mock_extract):
        mock_extract.return_value = [
            {'segments': [], 'num_segments': 0, 'audio_path': 'test1.wav'}
        ]
        
        with patch('builtins.print'):
            result = self.system.collect_features(['test1.wav'], ['male', 'female'])
            self.assertEqual(len(result), 1)
    
    @patch.object(GenderIdentificationSystem, 'collect_features')
    def test_analyze_training_data(self, mock_collect):
        mock_collect.return_value = [
            {
                'segments': [
                    {'mean_pitch': 120.0, 'pitch_std': 8.0},
                    {'mean_pitch': 130.0, 'pitch_std': 9.0}
                ],
                'true_label': 'male'
            },
            {
                'segments': [
                    {'mean_pitch': 200.0, 'pitch_std': 15.0},
                    {'mean_pitch': 190.0, 'pitch_std': 14.0}
                ],
                'true_label': 'female'
            }
        ]
        
        with patch('builtins.print'):
            self.system.analyze_training_data(['male1.wav', 'female1.wav'], ['male', 'female'])
        
        self.assertEqual(len(self.system.training_data), 2)
    
    @patch.object(GenderIdentificationSystem, 'collect_features')
    def test_analyze_training_data_no_male_female(self, mock_collect):
        mock_collect.return_value = [
            {
                'segments': [],
                'true_label': 'other'
            }
        ]
        
        with patch('builtins.print'):
            self.system.analyze_training_data(['other.wav'], ['other'])
        
        self.assertEqual(len(self.system.training_data), 1)
    
    @patch.object(GenderIdentificationSystem, 'collect_features')
    @patch.object(GenderClassifier, 'classify')
    def test_test_samples_with_labels(self, mock_classify, mock_collect):
        mock_collect.return_value = [
            {
                'segments': [{'mean_pitch': 200.0, 'pitch_std': 15.0}],
                'audio_path': 'test1.wav',
                'true_label': 'female'
            }
        ]
        mock_classify.return_value = {
            'final_prediction': 'female',
            'method': 'custom_rule_based',
            'num_segments': 1,
            'male_votes': 0,
            'female_votes': 1,
            'confidence': 1.0,
            'segment_predictions': ['female'],
            'segment_scores': [0.5]
        }
        
        with patch('builtins.print'):
            results = self.system.test_samples(['test1.wav'], ['female'])
        
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]['correct'])
        self.assertEqual(results[0]['true_label'], 'female')
    
    @patch.object(GenderIdentificationSystem, 'collect_features')
    @patch.object(GenderClassifier, 'classify')
    def test_test_samples_no_labels(self, mock_classify, mock_collect):
        mock_collect.return_value = [
            {
                'segments': [{'mean_pitch': 200.0, 'pitch_std': 15.0}],
                'audio_path': 'test1.wav'
            }
        ]
        mock_classify.return_value = {
            'final_prediction': 'female',
            'method': 'custom_rule_based',
            'num_segments': 1,
            'male_votes': 0,
            'female_votes': 1,
            'confidence': 1.0
        }
        
        with patch('builtins.print'):
            results = self.system.test_samples(['test1.wav'])
        
        self.assertEqual(len(results), 1)
        self.assertNotIn('true_label', results[0])
        self.assertNotIn('correct', results[0])
    
    @patch.object(GenderIdentificationSystem, 'collect_features')
    @patch.object(GenderClassifier, 'classify')
    def test_test_samples_accuracy_calculation(self, mock_classify, mock_collect):
        mock_collect.return_value = [
            {
                'segments': [{'mean_pitch': 200.0, 'pitch_std': 15.0}],
                'audio_path': 'test1.wav',
                'true_label': 'female'
            },
            {
                'segments': [{'mean_pitch': 120.0, 'pitch_std': 8.0}],
                'audio_path': 'test2.wav',
                'true_label': 'male'
            }
        ]
        mock_classify.side_effect = [
            {
                'final_prediction': 'female',
                'method': 'custom_rule_based',
                'num_segments': 1,
                'male_votes': 0,
                'female_votes': 1,
                'confidence': 1.0
            },
            {
                'final_prediction': 'male',
                'method': 'custom_rule_based',
                'num_segments': 1,
                'male_votes': 1,
                'female_votes': 0,
                'confidence': 1.0
            }
        ]
        
        with patch('builtins.print'):
            results = self.system.test_samples(['test1.wav', 'test2.wav'], ['female', 'male'])
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['correct'] for r in results))
    
    def test_save_results(self):
        results = [{'test': 'data'}]
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            with patch('builtins.print'):
                self.system.save_results(results, temp_path)
            
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            self.assertEqual(loaded, results)
        finally:
            os.unlink(temp_path)
    
    def test_save_results_csv(self):
        results = [{'field1': 'value1', 'field2': 'value2'}]
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            with patch('builtins.print'):
                self.system.save_results_csv(results, temp_path)
            
            self.assertTrue(os.path.exists(temp_path))
        finally:
            os.unlink(temp_path)
    
    def test_save_results_csv_empty(self):
        with patch('builtins.print'):
            self.system.save_results_csv([], "test.csv")
    
    def test_get_classifier_info(self):
        info = self.system.get_classifier_info()
        self.assertEqual(info['method'], 'custom_rule_based')
        self.assertIn('pitch_threshold', info)
        self.assertIn('mean_pitch_weight', info)
        self.assertIn('pitch_std_weight', info)

