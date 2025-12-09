import unittest
import sys
import os
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code_Logic'))
from src.Code_Logic.extract_voice_metadata import VoiceFeatureExtractor
#test the voice feature extractor
class TestVoiceFeatureExtractor(unittest.TestCase):
    #setup the extractor
    def setUp(self):
        self.extractor = VoiceFeatureExtractor()
    #test the default initialization
    def test_init_default(self):
        #test the default initialization
        self.assertEqual(self.extractor.sample_rate, 22050)
    #test the custom initialization
    def test_init_custom(self):
        extractor = VoiceFeatureExtractor(sample_rate=44100)
        #test the custom initialization is correct
        self.assertEqual(extractor.sample_rate, 44100)
    #test the load audio method with success
    @patch('librosa.load')
    def test_load_audio_success(self, mock_load):
        mock_audio = np.array([0.1, 0.2, 0.3])
        #test the load audio method with success
        mock_sr = 22050
        #test the load audio method with success
        mock_load.return_value = (mock_audio, mock_sr)

        audio, sr = self.extractor.load_audio("test.wav")
        #test the load audio method with success
        self.assertTrue(np.array_equal(audio, mock_audio))
        #test the load audio method with success    
        self.assertEqual(sr, mock_sr)
    #test the load audio method with error
    @patch('librosa.load')
    def test_load_audio_error(self, mock_load):
        #test the load audio method with error
        mock_load.side_effect = Exception("File not found")
        #test the load audio method with error
        with self.assertRaises(ValueError) as context:
            self.extractor.load_audio("nonexistent.wav")
        #test the load audio method with error
        self.assertIn("Error loading audio file", str(context.exception))

    @patch('librosa.pyin')
    def test_extract_fundamental_frequency(self, mock_pyin):
        mock_f0 = np.array([100.0, 150.0, 200.0, np.nan, 180.0])
        #test the extract fundamental frequency method with success
        mock_pyin.return_value = (mock_f0, None, None)
        #test the extract fundamental frequency method with success
        result = self.extractor.extract_fundamental_frequency(np.array([0.1, 0.2]))
        expected = np.array([100.0, 150.0, 200.0, 180.0])
        #test the extract fundamental frequency method with success
        self.assertTrue(np.array_equal(result, expected))

    @patch('librosa.pyin')
    def test_extract_fundamental_frequency_all_nan(self, mock_pyin):
        mock_f0 = np.array([np.nan, np.nan])
        #test the extract fundamental frequency method with all nan
        mock_pyin.return_value = (mock_f0, None, None)
        #test the extract fundamental frequency method with all nan
        result = self.extractor.extract_fundamental_frequency(np.array([0.1, 0.2]))
        #test the extract fundamental frequency method with all nan
        self.assertEqual(len(result), 0)

    def test_extract_segment_features_short_audio(self):
        short_audio = np.zeros(1000)
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_segment_features(short_audio, segment_duration=1.0)
        #test the extract segment features method with short audio
        self.assertIn("too short", str(context.exception))

    @patch.object(VoiceFeatureExtractor, 'extract_fundamental_frequency')
    def test_extract_segment_features_success(self, mock_extract_f0):
        #test the extract segment features method with success              
        mock_extract_f0.return_value = np.array([150.0, 160.0, 170.0])
        audio = np.zeros(22050 * 3)
        #test the extract segment features method with success
        result = self.extractor.extract_segment_features(audio, segment_duration=1.0)
        #test the extract segment features method with success      
        self.assertEqual(len(result), 3)
        #test the extract segment features method with success      
        self.assertEqual(result[0]['segment_index'], 0)
        #test the extract segment features method with success              
        self.assertIn('mean_pitch', result[0])
        #test the extract segment features method with success              
        self.assertIn('pitch_std', result[0])
    #test the extract features method with segments
    @patch.object(VoiceFeatureExtractor, 'load_audio')
    @patch.object(VoiceFeatureExtractor, 'extract_segment_features')
    @patch.object(VoiceFeatureExtractor, 'extract_fundamental_frequency')
    @patch.object(VoiceFeatureExtractor, 'extract_spectral_features')
    def test_extract_features_segments(self, mock_spectral, mock_f0, mock_segments, mock_load):
        #test the extract features method with segments
        mock_load.return_value = (np.zeros(22050 * 15), 22050)
        #test the extract features method with segments
        mock_segments.return_value = [
            {'segment_index': 0, 'mean_pitch': 150.0, 'pitch_std': 10.0},
            {'segment_index': 1, 'mean_pitch': 160.0, 'pitch_std': 12.0}
        ]
        #test the extract features method with segments
        mock_f0.return_value = np.array([150.0, 160.0, 170.0])
        #test the extract features method with segments
        mock_spectral.return_value = {'average_frequency': 150.0, 'peak_frequency': 200.0, 'frequency_std': 10.0}
        #test the extract features method with segments
        result = self.extractor.extract_features("test.wav", use_segments=True)
        #test the extract features method with segments
        self.assertIn('segments', result)
        #test the extract features method with segments
        self.assertIn('num_segments', result)
        #test the extract features method with segments
        self.assertEqual(result['num_segments'], 2)

    @patch.object(VoiceFeatureExtractor, 'load_audio')
    def test_extract_features_too_short(self, mock_load):
        #test the extract features method with too short audio
        mock_load.return_value = (np.zeros(22050 * 5), 22050)
        #test the extract features method with too short audio  
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_features("short.wav", use_segments=True)
        #test the extract features method with too short audio
        self.assertIn("at least", str(context.exception))

    @patch.object(VoiceFeatureExtractor, 'load_audio')
    @patch.object(VoiceFeatureExtractor, 'extract_segment_features')
    def test_extract_features_no_segments(self, mock_segments, mock_load):
        #test the extract features method with no segments
        mock_load.return_value = (np.zeros(22050 * 15), 22050)
        #test the extract features method with no segments
        mock_segments.return_value = []
        #test the extract features method with no segments
        with self.assertRaises(ValueError) as context:
            self.extractor.extract_features("test.wav", use_segments=True)
        #test the extract features method with no segments
        self.assertIn("No valid segments", str(context.exception))

    @patch.object(VoiceFeatureExtractor, 'load_audio')
    @patch.object(VoiceFeatureExtractor, 'extract_fundamental_frequency')
    def test_extract_features_legacy_mode(self, mock_f0, mock_load):
        #test the extract features method with legacy mode
        mock_load.return_value = (np.zeros(22050 * 15), 22050)

        mock_f0.return_value = np.array([100.0, 150.0, 200.0])
        #test the extract features method with legacy mode
        result = self.extractor.extract_features("test.wav", use_segments=False)
        #test the extract features method with legacy mode
        self.assertIn('average_frequency', result)
        #test the extract features method with legacy mode
        self.assertIn('peak_frequency', result)
        #test the extract features method with legacy mode
        self.assertIn('frequency_std', result)
    #test the extract features method with batch success
    @patch.object(VoiceFeatureExtractor, 'extract_features')
    def test_extract_features_batch_success(self, mock_extract):
        def side_effect(path, use_segments=True):
            #test the extract features method with batch success    
            return {'segments': [], 'num_segments': 0}
        #test the extract features method with batch success
        mock_extract.side_effect = side_effect
        #test the extract features method with batch success    
        result = self.extractor.extract_features_batch(["file1.wav", "file2.wav"])
        #test the extract features method with batch success    
        self.assertEqual(len(result), 2)
        #test the extract features method with batch success    
        self.assertEqual(result[0]['audio_path'], "file1.wav")
    #test the extract features method with batch with error
    @patch.object(VoiceFeatureExtractor, 'extract_features')
    def test_extract_features_batch_with_error(self, mock_extract):
        def side_effect(path, use_segments=True):
            if path == "bad.wav":
                raise ValueError("Error")
            return {'segments': [], 'num_segments': 0}
        #test the extract features method with batch with error     
        mock_extract.side_effect = side_effect
        #test the extract features method with batch with error     
        result = self.extractor.extract_features_batch(["file1.wav", "bad.wav", "file2.wav"])
        #test the extract features method with batch with error     
        self.assertEqual(len(result), 2)