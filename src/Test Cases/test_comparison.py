import unittest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code Logic'))
from main import GenderIdentificationSystem
from classification import GenderClassifier


class TestVoiceSampleComparison(unittest.TestCase):
    
    def setUp(self):
        self.system = GenderIdentificationSystem(
            pitch_threshold=165.0,
            mean_pitch_weight=0.7,
            pitch_std_weight=0.3
        )
        
        self.expected_results = {
            'male_sample_1.wav': {
                'final_prediction': 'male',
                'num_segments': 10,
                'male_votes': 7,
                'female_votes': 3,
                'confidence': 0.7,
                'true_label': 'male',
                'correct': True
            },
            'female_sample_1.wav': {
                'final_prediction': 'female',
                'num_segments': 12,
                'male_votes': 2,
                'female_votes': 10,
                'confidence': 0.833,
                'true_label': 'female',
                'correct': True
            },
            'male_sample_2.wav': {
                'final_prediction': 'male',
                'num_segments': 15,
                'male_votes': 11,
                'female_votes': 4,
                'confidence': 0.733,
                'true_label': 'male',
                'correct': True
            },
            'female_sample_2.wav': {
                'final_prediction': 'female',
                'num_segments': 11,
                'male_votes': 3,
                'female_votes': 8,
                'confidence': 0.727,
                'true_label': 'female',
                'correct': True
            }
        }
    
    @patch.object(GenderIdentificationSystem, 'collect_features')
    @patch.object(GenderClassifier, 'classify')
    def test_voice_sample_comparison_report(self, mock_classify, mock_collect):
        mock_collect.return_value = [
            {
                'segments': [{'mean_pitch': 120.0, 'pitch_std': 8.0}] * 10,
                'audio_path': 'male_sample_1.wav',
                'true_label': 'male'
            },
            {
                'segments': [{'mean_pitch': 200.0, 'pitch_std': 15.0}] * 12,
                'audio_path': 'female_sample_1.wav',
                'true_label': 'female'
            },
            {
                'segments': [{'mean_pitch': 130.0, 'pitch_std': 9.0}] * 15,
                'audio_path': 'male_sample_2.wav',
                'true_label': 'male'
            },
            {
                'segments': [{'mean_pitch': 190.0, 'pitch_std': 14.0}] * 11,
                'audio_path': 'female_sample_2.wav',
                'true_label': 'female'
            }
        ]
        
        def classify_side_effect(features):
            segments = features['segments']
            num_segments = len(segments)
            mean_pitch = segments[0]['mean_pitch']
            
            if mean_pitch < 165.0:
                male_votes = num_segments
                female_votes = 0
                prediction = 'male'
            else:
                male_votes = 0
                female_votes = num_segments
                prediction = 'female'
            
            confidence = max(male_votes, female_votes) / num_segments
            
            return {
                'final_prediction': prediction,
                'method': 'custom_rule_based',
                'num_segments': num_segments,
                'male_votes': male_votes,
                'female_votes': female_votes,
                'confidence': confidence,
                'segment_predictions': [prediction] * num_segments,
                'segment_scores': [0.5 if prediction == 'female' else -0.5] * num_segments
            }
        
        mock_classify.side_effect = classify_side_effect
        
        test_paths = list(self.expected_results.keys())
        test_labels = [self.expected_results[p]['true_label'] for p in test_paths]
        
        with patch('builtins.print'):
            actual_results = self.system.test_samples(test_paths, test_labels)
        
        comparison_report = []
        all_match = True
        
        for actual in actual_results:
            audio_path = actual['audio_path']
            expected = self.expected_results.get(audio_path)
            
            if expected:
                comparison = {
                    'audio_file': audio_path,
                    'expected_prediction': expected['final_prediction'],
                    'actual_prediction': actual['final_prediction'],
                    'prediction_match': expected['final_prediction'] == actual['final_prediction'],
                    'expected_segments': expected['num_segments'],
                    'actual_segments': actual['num_segments'],
                    'segments_match': expected['num_segments'] == actual['num_segments'],
                    'expected_male_votes': expected['male_votes'],
                    'actual_male_votes': actual['male_votes'],
                    'male_votes_match': expected['male_votes'] == actual['male_votes'],
                    'expected_female_votes': expected['female_votes'],
                    'actual_female_votes': actual['female_votes'],
                    'female_votes_match': expected['female_votes'] == actual['female_votes'],
                    'expected_confidence': expected['confidence'],
                    'actual_confidence': actual['confidence'],
                    'confidence_match': abs(expected['confidence'] - actual['confidence']) < 0.1,
                    'expected_correct': expected['correct'],
                    'actual_correct': actual.get('correct', False),
                    'correct_match': expected['correct'] == actual.get('correct', False)
                }
                
                comparison_report.append(comparison)
                
                if not comparison['prediction_match']:
                    all_match = False
        
        print("\n" + "="*80)
        print("VOICE SAMPLE COMPARISON REPORT")
        print("="*80)
        print(f"\nTotal Samples Tested: {len(comparison_report)}")
        print(f"All Predictions Match: {all_match}")
        print("\nDetailed Comparison:")
        print("-" * 80)
        
        for comp in comparison_report:
            print(f"\nFile: {comp['audio_file']}")
            print(f"  Prediction: Expected={comp['expected_prediction']}, "
                  f"Actual={comp['actual_prediction']}, "
                  f"Match={comp['prediction_match']}")
            print(f"  Segments: Expected={comp['expected_segments']}, "
                  f"Actual={comp['actual_segments']}, "
                  f"Match={comp['segments_match']}")
            print(f"  Male Votes: Expected={comp['expected_male_votes']}, "
                  f"Actual={comp['actual_male_votes']}, "
                  f"Match={comp['male_votes_match']}")
            print(f"  Female Votes: Expected={comp['expected_female_votes']}, "
                  f"Actual={comp['actual_female_votes']}, "
                  f"Match={comp['female_votes_match']}")
            print(f"  Confidence: Expected={comp['expected_confidence']:.3f}, "
                  f"Actual={comp['actual_confidence']:.3f}, "
                  f"Match={comp['confidence_match']}")
            print(f"  Correct: Expected={comp['expected_correct']}, "
                  f"Actual={comp['actual_correct']}, "
                  f"Match={comp['correct_match']}")
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        prediction_matches = sum(1 for c in comparison_report if c['prediction_match'])
        segments_matches = sum(1 for c in comparison_report if c['segments_match'])
        votes_matches = sum(1 for c in comparison_report if c['male_votes_match'] and c['female_votes_match'])
        confidence_matches = sum(1 for c in comparison_report if c['confidence_match'])
        correct_matches = sum(1 for c in comparison_report if c['correct_match'])
        
        print(f"Prediction Matches: {prediction_matches}/{len(comparison_report)} "
              f"({100*prediction_matches/len(comparison_report):.1f}%)")
        print(f"Segments Matches: {segments_matches}/{len(comparison_report)} "
              f"({100*segments_matches/len(comparison_report):.1f}%)")
        print(f"Votes Matches: {votes_matches}/{len(comparison_report)} "
              f"({100*votes_matches/len(comparison_report):.1f}%)")
        print(f"Confidence Matches: {confidence_matches}/{len(comparison_report)} "
              f"({100*confidence_matches/len(comparison_report):.1f}%)")
        print(f"Correct Matches: {correct_matches}/{len(comparison_report)} "
              f"({100*correct_matches/len(comparison_report):.1f}%)")
        
        self.assertIsNotNone(comparison_report)
        self.assertEqual(len(comparison_report), len(self.expected_results))

