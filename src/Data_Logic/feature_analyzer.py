import numpy as np
from typing import List, Dict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Code_Logic'))
from src.Code_Logic.extract_voice_metadata import VoiceFeatureExtractor
from utils import save_to_csv
class FeatureAnalyzer:#feature analyzer class used to analyze the features of the audio paths and labels
    def __init__(self):#initialize the feature analyzer
        self.extractor = VoiceFeatureExtractor()#initialize the voice feature extractor
        self.results = []#initialize the results list
    def analyze_batch(self, audio_paths: List[str], labels: List[str]):#analyze the batch of audio paths and labels
        for audio_path, label in zip(audio_paths, labels):#loop through the audio paths and labels
            try:#analyze the batch of audio paths and labels
                features = self.extractor.extract_features(audio_path, use_segments=False)#extract the features from the audio path
                result = {#create a new entry for the results
                    'audio_path': audio_path,
                    'label': label,
                    'average_frequency': features.get('average_frequency', 0.0),
                    'peak_frequency': features.get('peak_frequency', 0.0),
                    'frequency_std': features.get('frequency_std', 0.0)
                }
                self.results.append(result)
            except Exception as e:
                print(f"Error analyzing {audio_path}: {e}")
    def save_results_csv(self, output_path: str):#save the results to a csv file
        if self.results:#if the results are not empty, then:
            save_to_csv(self.results, output_path)#save the results to a csv file

