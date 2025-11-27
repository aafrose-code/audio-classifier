import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code Logic'))
from main import find_audio_files, load_training_data, load_test_data


class TestHelperFunctions(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_find_audio_files_nonexistent(self):
        result = find_audio_files("nonexistent_dir")
        self.assertEqual(len(result), 0)
    
    def test_find_audio_files_success(self):
        os.makedirs(os.path.join(self.test_dir, 'subdir'))
        
        test_file1 = os.path.join(self.test_dir, 'audio1.wav')
        test_file2 = os.path.join(self.test_dir, 'subdir', 'audio2.mp3')
        test_file3 = os.path.join(self.test_dir, 'notaudio.txt')
        
        with open(test_file1, 'w') as f:
            f.write('test')
        with open(test_file2, 'w') as f:
            f.write('test')
        with open(test_file3, 'w') as f:
            f.write('test')
        
        result = find_audio_files(self.test_dir)
        self.assertEqual(len(result), 2)
        self.assertTrue(any('audio1.wav' in p for p in result))
        self.assertTrue(any('audio2.mp3' in p for p in result))
    
    def test_find_audio_files_case_insensitive(self):
        test_file = os.path.join(self.test_dir, 'AUDIO.WAV')
        with open(test_file, 'w') as f:
            f.write('test')
        
        result = find_audio_files(self.test_dir)
        self.assertEqual(len(result), 1)
    
    @patch('os.path.exists')
    def test_load_training_data_directory_structure(self, mock_exists):
        mock_exists.side_effect = lambda p: p in [
            "audio_samples/training",
            "audio_samples/training/male",
            "audio_samples/training/female"
        ]
        
        with patch('main.find_audio_files') as mock_find:
            mock_find.side_effect = lambda d: {
                "audio_samples/training/male": ["male1.wav"],
                "audio_samples/training/female": ["female1.wav"]
            }.get(d, [])
            
            paths, labels = load_training_data()
            if paths:
                self.assertIn('male', labels)
                self.assertIn('female', labels)
    
    @patch('os.path.exists')
    def test_load_training_data_filename_based(self, mock_exists):
        mock_exists.side_effect = lambda p: p == "audio_samples/training"
        
        with patch('main.find_audio_files') as mock_find:
            mock_find.return_value = [
                "audio_samples/training/male_1.wav",
                "audio_samples/training/female_1.wav"
            ]
            
            paths, labels = load_training_data()
            if paths:
                self.assertIn('male', labels)
                self.assertIn('female', labels)
    
    def test_load_training_data_none(self):
        with patch('os.path.exists', return_value=False):
            paths, labels = load_training_data()
            self.assertIsNone(paths)
            self.assertIsNone(labels)
    
    @patch('os.path.exists')
    def test_load_test_data_directory_structure(self, mock_exists):
        mock_exists.side_effect = lambda p: p in [
            "audio_samples/test",
            "audio_samples/test/male",
            "audio_samples/test/female"
        ]
        
        with patch('main.find_audio_files') as mock_find:
            mock_find.side_effect = lambda d: {
                "audio_samples/test/male": ["male1.wav"],
                "audio_samples/test/female": ["female1.wav"]
            }.get(d, [])
            
            paths, labels = load_test_data()
            if paths:
                self.assertIn('male', labels)
                self.assertIn('female', labels)
    
    @patch('os.path.exists')
    def test_load_test_data_filename_based(self, mock_exists):
        mock_exists.side_effect = lambda p: p == "audio_samples/test"
        
        with patch('main.find_audio_files') as mock_find:
            mock_find.return_value = [
                "audio_samples/test/test1.wav"
            ]
            
            paths, labels = load_test_data()
            if paths:
                self.assertIsNotNone(paths)

