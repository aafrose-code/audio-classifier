import numpy as np
from typing import Dict, List, Tuple, Optional


class CustomRuleBasedClassifier:
    
    def __init__(self, 
                 pitch_threshold: float = 165.0,
                 mean_pitch_weight: float = 0.7,
                 pitch_std_weight: float = 0.3):
        self.pitch_threshold = pitch_threshold
        self.mean_pitch_weight = mean_pitch_weight
        self.pitch_std_weight = pitch_std_weight
        
        total_weight = mean_pitch_weight + pitch_std_weight
        if total_weight > 0:
            self.mean_pitch_weight = mean_pitch_weight / total_weight
            self.pitch_std_weight = pitch_std_weight / total_weight
    
    def compute_segment_score(self, mean_pitch: float, pitch_std: float) -> float:
        mean_pitch_score = (mean_pitch - self.pitch_threshold) / self.pitch_threshold
        
        pitch_std_score = pitch_std / 50.0
        
        weighted_score = (self.mean_pitch_weight * mean_pitch_score + 
                         self.pitch_std_weight * pitch_std_score)
        
        return weighted_score
    
    def classify_segment(self, mean_pitch: float, pitch_std: float) -> str:
        score = self.compute_segment_score(mean_pitch, pitch_std)
        
        if score >= 0:
            return 'female'
        else:
            return 'male'
    
    def classify(self, features: Dict) -> Dict[str, any]:
        if 'segments' not in features:
            raise ValueError("Features must contain 'segments' list for segment-based classification")
        
        segments = features['segments']
        
        if len(segments) == 0:
            raise ValueError("No segments found in features")
        
        segment_predictions = []
        segment_scores = []
        
        for segment in segments:
            mean_pitch = segment['mean_pitch']
            pitch_std = segment['pitch_std']
            
            score = self.compute_segment_score(mean_pitch, pitch_std)
            prediction = self.classify_segment(mean_pitch, pitch_std)
            
            segment_predictions.append(prediction)
            segment_scores.append(score)
        
        male_votes = sum(1 for pred in segment_predictions if pred == 'male')
        female_votes = sum(1 for pred in segment_predictions if pred == 'female')
        
        if female_votes > male_votes:
            final_prediction = 'female'
        elif male_votes > female_votes:
            final_prediction = 'male'
        else:
            avg_score = np.mean(segment_scores)
            final_prediction = 'female' if avg_score >= 0 else 'male'
        
        total_votes = len(segment_predictions)
        confidence = max(male_votes, female_votes) / total_votes if total_votes > 0 else 0.0
        
        return {
            'prediction': final_prediction,
            'segment_predictions': segment_predictions,
            'segment_scores': segment_scores,
            'male_votes': male_votes,
            'female_votes': female_votes,
            'confidence': confidence,
            'num_segments': len(segments)
        }
    
    def classify_batch(self, features_list: List[Dict]) -> List[Dict[str, any]]:
        return [self.classify(features) for features in features_list]


class GenderClassifier:
    
    def __init__(self, pitch_threshold: float = 165.0,
                 mean_pitch_weight: float = 0.7,
                 pitch_std_weight: float = 0.3):
        self.classifier = CustomRuleBasedClassifier(
            pitch_threshold=pitch_threshold,
            mean_pitch_weight=mean_pitch_weight,
            pitch_std_weight=pitch_std_weight
        )
    
    def classify(self, features: Dict) -> Dict[str, any]:
        result = self.classifier.classify(features)
        
        result['final_prediction'] = result['prediction']
        result['method'] = 'custom_rule_based'
        
        return result
