import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code_Logic'))
from src.Code_Logic.main import find_audio_files, load_training_data, load_test_data
#test the helper functions
class TestHelperFunctions(unittest.TestCase):
    #setup the test directory
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    #tear down the test directory
    def tearDown(self):
        #remove the test directory
        shutil.rmtree(self.test_dir)

    def test_find_audio_files_nonexistent(self):
        result = find_audio_files("nonexistent_dir")
        #test the result is correct
        self.assertEqual(len(result), 0)

    def test_find_audio_files_success(self):
        os.makedirs(os.path.join(self.test_dir, 'subdir'))
        #test the find audio files method with success
        test_file1 = os.path.join(self.test_dir, 'audio1.wav')
        test_file2 = os.path.join(self.test_dir, 'subdir', 'audio2.mp3')
        test_file3 = os.path.join(self.test_dir, 'notaudio.txt')
        #test the find audio files method with success
        with open(test_file1, 'w') as f:
            f.write('test')
        with open(test_file2, 'w') as f:
            f.write('test')
        with open(test_file3, 'w') as f:
            f.write('test')
        #test the find audio files method with success
        result = find_audio_files(self.test_dir)
        #test the result is correct
        self.assertEqual(len(result), 2)
        #test the result is correct
        self.assertTrue(any('audio1.wav' in p for p in result))
        #test the result is correct
        self.assertTrue(any('audio2.mp3' in p for p in result))
    #test the find audio files method with case insensitive
    def test_find_audio_files_case_insensitive(self):
        test_file = os.path.join(self.test_dir, 'AUDIO.WAV')
        with open(test_file, 'w') as f:
            f.write('test')
        #test the find audio files method with case insensitive
        result = find_audio_files(self.test_dir)
        #test the result is correct
        self.assertEqual(len(result), 1)

    def test_load_training_data_none(self):
        #test the load training data method with none
        with patch('os.path.exists', return_value=False):
            paths, labels = load_training_data()
            #test the result is correct
            self.assertIsNone(paths)
            #test the result is correct
            self.assertIsNone(labels)#test the result is correct