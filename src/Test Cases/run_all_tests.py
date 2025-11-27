import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Code Logic'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_feature_extraction import TestVoiceFeatureExtractor
from test_classification import TestCustomRuleBasedClassifier, TestGenderClassifier
from test_system import TestGenderIdentificationSystem
from test_helpers import TestHelperFunctions
from test_comparison import TestVoiceSampleComparison

def create_test_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestVoiceFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomRuleBasedClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestGenderClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestGenderIdentificationSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestVoiceSampleComparison))
    
    return suite

if __name__ == '__main__':
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

