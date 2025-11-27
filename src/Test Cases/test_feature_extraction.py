import unittest
import sys
import os
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code Logic'))
from extract_voice_metadata import VoiceFeatureExtractor


class TestVoiceFeatureExtractor(unittest.TestCase):
    
    def setUp(self):
        self.extractor = VoiceFeatureExtractor()
    
    def test_init_default(self):
        self.assertEqual(self.extractor.sample_rate, 22050)
    
    def test_init_custom(self):
        extractor = VoiceFeatureExtractor(sample_rate=44100)
        self.assertEqual(extractor.sample_rate, 44100)
    
    @patch('librosa.load')
    def test_load_audio_success(self, mock_load):
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_sr = 22050
        mock_load.return_value = (mock_audio, mock_sr)
        
        audio, sr = self.extractor.load_audio("test.wav")
        self.assertTrue(np.array_equal(audio, mock_audio))
        self.assertEqual(sr, mock_sr)
    
    @patch('librosa.load')
    def test_load_audio_error(self, mock_load):
        mock_load.side_effect = Exception("File not found")
        
        with self.assertRaises(ValueError) as context:
            self.extractor.load_audio("nonexistent.wav")
        self.assertIn("Error loading audio file", str(context.exception))
    
    @patch('librosa.pyin')
    def test_extract_fundamental_frequency(self, mock_pyin):
        mock_f0 = np.array([100.0, 150.0, 200.0, np.nan, 180.0])
        mock_pyin.return_value = (mock_f0, None, None)
        
        result = self.extractor.extract_fundamental_frequency(np.array([0.1, 0.2]))
        expected = np.array([100.0, 150.0, 200.0, 180.0])
        self.assertTrue(np.array_equal(result, expected))
    
    @patch('librosa.pyin')
    def test_extract_fundamental_frequency_all_nan(self, mock_pyin):
        mock_f0 = np.array([np.nan, np.nan])
        mock_pyin.return_value = (mock_f0, None, None)
        
        result = self.extractor.extract_fundamental_frequency(np.array([0.1, 0.2]))
        self.assertEqual(len(result), 0)
    
    def test_extract_segment_features_short_audio(self):
        short_audio = np.zeros(1000)
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_segment_features(short_audio, segment_duration=1.0)
        self.assertIn("too short", str(context.exception))
    
    @patch.object(VoiceFeatureExtractor, 'extract_fundamental_frequency')
    def test_extract_segment_features_success(self, mock_extract_f0):
        mock_extract_f0.return_value = np.array([150.0, 160.0, 170.0])
        audio = np.zeros(22050 * 3)
        
        result = self.extractor.extract_segment_features(audio, segment_duration=1.0)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['segment_index'], 0)
        self.assertIn('mean_pitch', result[0])
        self.assertIn('pitch_std', result[0])
    
    @patch.object(VoiceFeatureExtractor, 'extract_fundamental_frequency')
    def test_extract_segment_features_empty_f0(self, mock_extract_f0):
        mock_extract_f0.return_value = np.array([])
        audio = np.zeros(22050 * 2)
        
        result = self.extractor.extract_segment_features(audio, segment_duration=1.0)
        self.assertEqual(len(result), 0)
    
    @patch.object(VoiceFeatureExtractor, 'load_audio')
    @patch.object(VoiceFeatureExtractor, 'extract_segment_features')
    def test_extract_features_segments(self, mock_segments, mock_load):
        mock_load.return_value = (np.zeros(22050 * 15), 22050)
        mock_segments.return_value = [
            {'segment_index': 0, 'mean_pitch': 150.0, 'pitch_std': 10.0},
            {'segment_index': 1, 'mean_pitch': 160.0, 'pitch_std': 12.0}
        ]
        
        result = self.extractor.extract_features("test.wav", use_segments=True)
        self.assertIn('segments', result)
        self.assertIn('num_segments', result)
        self.assertEqual(result['num_segments'], 2)
    
    @patch.object(VoiceFeatureExtractor, 'load_audio')
    def test_extract_features_too_short(self, mock_load):
        mock_load.return_value = (np.zeros(22050 * 5), 22050)
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_features("short.wav", use_segments=True)
        self.assertIn("at least", str(context.exception))
    
    @patch.object(VoiceFeatureExtractor, 'load_audio')
    @patch.object(VoiceFeatureExtractor, 'extract_segment_features')
    def test_extract_features_no_segments(self, mock_segments, mock_load):
        mock_load.return_value = (np.zeros(22050 * 15), 22050)
        mock_segments.return_value = []
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_features("test.wav", use_segments=True)
        self.assertIn("No valid segments", str(context.exception))
    
    @patch.object(VoiceFeatureExtractor, 'load_audio')
    @patch.object(VoiceFeatureExtractor, 'extract_fundamental_frequency')
    def test_extract_features_no_segments_legacy(self, mock_f0, mock_load):
        mock_load.return_value = (np.zeros(22050 * 15), 22050)
        mock_f0.return_value = np.array([])
        
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_features("test.wav", use_segments=False)
        self.assertIn("No valid fundamental frequency", str(context.exception))
    
    @patch.object(VoiceFeatureExtractor, 'load_audio')
    @patch.object(VoiceFeatureExtractor, 'extract_fundamental_frequency')
    def test_extract_features_legacy_mode(self, mock_f0, mock_load):
        mock_load.return_value = (np.zeros(22050 * 15), 22050)
        mock_f0.return_value = np.array([100.0, 150.0, 200.0])
        
        result = self.extractor.extract_features("test.wav", use_segments=False)
        self.assertIn('average_frequency', result)
        self.assertIn('peak_frequency', result)
        self.assertIn('frequency_std', result)
    
    @patch.object(VoiceFeatureExtractor, 'extract_features')
    def test_extract_features_batch_success(self, mock_extract):
        mock_extract.return_value = {'segments': [], 'num_segments': 0}
        
        result = self.extractor.extract_features_batch(["file1.wav", "file2.wav"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['audio_path'], "file1.wav")
    
    @patch.object(VoiceFeatureExtractor, 'extract_features')
    def test_extract_features_batch_with_error(self, mock_extract):
        def side_effect(path):
            if path == "bad.wav":
                raise ValueError("Error")
            return {'segments': [], 'num_segments': 0}
        
        mock_extract.side_effect = side_effect
        
        result = self.extractor.extract_features_batch(["file1.wav", "bad.wav", "file2.wav"])
        self.assertEqual(len(result), 2)

