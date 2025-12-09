import unittest
import sys
import os
import tempfile
from unittest.mock import patch
#path to the code logic
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code_Logic'))
from src.Code_Logic.main import GenderIdentificationSystem
from src.Code_Logic.extract_voice_metadata import VoiceFeatureExtractor
from src.Code_Logic.classification import GenderClassifier
#test the gender identification system
class TestGenderIdentificationSystem(unittest.TestCase):
    #setup the system
    def setUp(self):
        self.system = GenderIdentificationSystem()
    #test the default initialization
    def test_init_default(self):#test the default initialization
        self.assertIsNotNone(self.system.feature_extractor)#test the feature extractor is not None
        self.assertIsNotNone(self.system.classifier)#test the classifier is not None
        self.assertEqual(len(self.system.training_data), 0)#test the training data is empty list
        self.assertEqual(len(self.system.test_data), 0)#test the test data is empty list
    #test the custom initialization
    def test_init_custom(self):#test the custom initialization
        system = GenderIdentificationSystem(
            pitch_threshold=180.0,
            mean_pitch_weight=0.8,
            pitch_std_weight=0.2
        )#initialize the system with the custom values
        self.assertIsNotNone(system.classifier)#test the classifier is not None
    @patch.object(VoiceFeatureExtractor, 'extract_features_batch')
    def test_collect_features_no_labels(self, mock_extract):
        #test the collect features method with no labels (mock the extract features method)
        mock_extract.return_value = [
            {'segments': [], 'num_segments': 0, 'audio_path': 'test1.wav'},
            {'segments': [], 'num_segments': 0, 'audio_path': 'test2.wav'}
        ]
        #test the collect features method with no labels (mock the extract features method)
        result = self.system.collect_features(['test1.wav', 'test2.wav'])
        #test the results are correct
        self.assertEqual(len(result), 2)
        #test the results are correct
        self.assertNotIn('true_label', result[0])
#test the collect features method with labels
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
        #test the test samples method with no labels (mock the classify method)
        mock_classify.return_value = {
            'final_prediction': 'female',
            'method': 'custom_rule_based',
            'num_segments': 1,
            'male_votes': 0,
            'female_votes': 1,
            'confidence': 1.0
        }
        #test the test samples method
        with patch('builtins.print'):
            results = self.system.test_samples(['test1.wav'])
        #test the results are correct
        self.assertEqual(len(results), 1)
        #test the results are correct
        self.assertNotIn('true_label', results[0])
        #test the results are correct
        self.assertNotIn('correct', results[0])

    @patch.object(GenderIdentificationSystem, 'collect_features')
    @patch.object(GenderClassifier, 'classify')
    def test_test_samples_accuracy_calculation(self, mock_classify, mock_collect):
        #test the test samples method
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
        #test the test samples method
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

        #test the test samples method
        with patch('builtins.print'):
            results = self.system.test_samples(['test1.wav', 'test2.wav'], ['female', 'male'])
        #test the results are correct
        self.assertEqual(len(results), 2)
        #test the results are correct
        self.assertTrue(all(r['correct'] for r in results))
    #test the save results csv method
    def test_save_results_csv(self):
        results = [{'field1': 'value1', 'field2': 'value2'}]
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        #test the save results csv method
        try:
            #test the save results csv method
            with patch('builtins.print'):
                self.system.save_results_csv(results, temp_path)#save the results to the temporary file
            #test the temporary file exists
            self.assertTrue(os.path.exists(temp_path))
        finally:
            #delete the temporary file
            os.unlink(temp_path)

    def test_get_classifier_info(self):
        info = self.system.get_classifier_info()
        #test the method
        self.assertEqual(info['method'], 'custom_rule_based')
        #test the pitch threshold
        self.assertIn('pitch_threshold', info)
        #test the pitch threshold
        self.assertIn('mean_pitch_weight', info)
        #test the mean pitch weight
        self.assertIn('pitch_std_weight', info)