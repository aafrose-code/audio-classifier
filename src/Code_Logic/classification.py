import numpy as np
from typing import Dict, List, Optional
from config import (DEFAULT_PITCH_THRESHOLD, DEFAULT_FEMALE_THRESHOLD, DEFAULT_MEAN_PITCH_WEIGHT,
                   DEFAULT_PITCH_STD_WEIGHT, DEFAULT_AVG_FREQ_WEIGHT, DEFAULT_PEAK_FREQ_WEIGHT, DEFAULT_FREQ_STD_WEIGHT,
                   DEFAULT_SPECTRAL_WEIGHT, USE_ADAPTIVE_THRESHOLD, ADAPTIVE_THRESHOLD_ALPHA)
##custom classifier class that implements a hybrid of rule and segment based classification
class CustomRuleBasedClassifier:
#initialize the classifier with the default values (adaptive threshold is also used to meet all population demographics)
#all of these default values are set in the config.py file just for organization purposes
    def __init__(self,
                 pitch_threshold: float = DEFAULT_PITCH_THRESHOLD,
                 female_threshold: float = DEFAULT_FEMALE_THRESHOLD,
                 mean_pitch_weight: float = DEFAULT_MEAN_PITCH_WEIGHT,
                 pitch_std_weight: float = DEFAULT_PITCH_STD_WEIGHT,
                 avg_freq_weight: float = DEFAULT_AVG_FREQ_WEIGHT,
                 peak_freq_weight: float = DEFAULT_PEAK_FREQ_WEIGHT,
                 freq_std_weight: float = DEFAULT_FREQ_STD_WEIGHT,
                 spectral_weight: float = DEFAULT_SPECTRAL_WEIGHT,
                 use_adaptive_threshold: bool = USE_ADAPTIVE_THRESHOLD,
                 adaptive_alpha: float = ADAPTIVE_THRESHOLD_ALPHA):
        self.base_pitch_threshold = pitch_threshold
        self.base_female_threshold = female_threshold
        self.pitch_threshold = pitch_threshold
        self.female_threshold = female_threshold
        self.mean_pitch_weight = mean_pitch_weight
        self.pitch_std_weight = pitch_std_weight
        self.avg_freq_weight = avg_freq_weight
        self.peak_freq_weight = peak_freq_weight
        self.freq_std_weight = freq_std_weight
        self.spectral_weight = spectral_weight
        self.use_adaptive_threshold = use_adaptive_threshold
        self.adaptive_alpha = adaptive_alpha
        self.training_male_pitches = []
        self.training_female_pitches = []
        self.adaptive_threshold_calculated = False
        #total weight of the segment based features is needed to normalize the weights
        total_segment_weight = mean_pitch_weight + pitch_std_weight
        if total_segment_weight > 0:#if the total weight is greater than 0, then:
            #normalize the weights so that the total weight is 1
            self.mean_pitch_weight = mean_pitch_weight / total_segment_weight
            self.pitch_std_weight = pitch_std_weight / total_segment_weight
        #total weight of the overall features is needed to normalize the weights
        total_overall_weight = avg_freq_weight + peak_freq_weight + freq_std_weight
        if total_overall_weight > 0:#if the total weight is greater than 0, then:
            #normalize the weights so that the total weight is 1
            self.avg_freq_weight = avg_freq_weight / total_overall_weight
            self.peak_freq_weight = peak_freq_weight / total_overall_weight
            self.freq_std_weight = freq_std_weight / total_overall_weight
#this definition is used to calibrate the thresholds from the training data
    def calibrate_thresholds_from_training(self, training_features: List[Dict]):
        #if the adaptive threshold is not used or the training data is not provided, then:
        if not self.use_adaptive_threshold or not training_features:
            #return the function
            return
        #initialize the lists to store the pitches for the male and female training data
        male_pitches = []
        female_pitches = []
        #iterate through the empty training data array and store the pitches for the male and female training data
        for features in training_features:
            if 'true_label' not in features:
                continue
            #get the label from the training data
            label = features['true_label'].lower()
            #iterate through the segments in the training data and store the pitches for the male and female training data
            for segment in features.get('segments', []):
                if label == 'male':
                    male_pitches.append(segment['mean_pitch'])
                elif label == 'female':
                    female_pitches.append(segment['mean_pitch'])
        #if the male and female pitches are not empty, then:
        if len(male_pitches) > 0 and len(female_pitches) > 0:
            #store the male and female pitches in the training data array
            self.training_male_pitches = male_pitches
            self.training_female_pitches = female_pitches
            #calculate the mean of the male and female pitches
            male_mean = np.mean(male_pitches)
            female_mean = np.mean(female_pitches)
            #calculate the threshold for the male and female pitches
            calculated_threshold = (male_mean + female_mean) / 2.0
            calculated_female_threshold = female_mean - (female_mean - calculated_threshold) * 0.5
            #calculate the final threshold for the male and female pitches
            self.pitch_threshold = (self.adaptive_alpha * calculated_threshold +
                                   (1 - self.adaptive_alpha) * self.base_pitch_threshold)
            self.female_threshold = (self.adaptive_alpha * calculated_female_threshold +
                                    (1 - self.adaptive_alpha) * self.base_female_threshold)
            #set the adaptive threshold calculated flag to true...this is basically used to check if the adaptive threshold is used in the classify function
            self.adaptive_threshold_calculated = True
            #print the results for the user to see
            print(f"\n  Adaptive Thresholds Calculated:")
            print(f"    Training Male Mean: {male_mean:.2f} Hz")
            print(f"    Training Female Mean: {female_mean:.2f} Hz")
            print(f"    Calculated Threshold: {calculated_threshold:.2f} Hz")
            print(f"    Final Pitch Threshold: {self.pitch_threshold:.2f} Hz (blended)")
            print(f"    Final Female Threshold: {self.female_threshold:.2f} Hz (blended)")
#this definition is used to compute the spectral score for the segment based classification
    def compute_spectral_score(self, segment: Dict) -> float:
        #initialize the score to 0
        score = 0.0
        #get the spectral centroid from the segment (part of the spectral analysis)
        centroid = segment.get('spectral_centroid', 0.0)
        if centroid > 0:

            centroid_score = (centroid - 2000.0) / 2000.0
            score += centroid_score * 0.15
        #get the zero crossing rate from the segment (part of the spectral analysis)
        zcr = segment.get('zero_crossing_rate', 0.0)
        if zcr > 0:
            #calculate the zero crossing rate score
            zcr_score = -(zcr - 0.05) / 0.05
            score += zcr_score * 0.1
        #get the first MFCC from the segment (part of the spectral analysis)
        mfcc_1 = segment.get('mfcc_1', 0.0)
        if mfcc_1 != 0:
            #calculate the MFCC score
            mfcc_score = -mfcc_1 / 500.0
            score += mfcc_score * 0.02
        #return the spectral score multiplied by the spectral weight
        return score * self.spectral_weight
#this definition is basically used to compute the segment score for the segment based classification
    def compute_segment_score(self, mean_pitch: float, pitch_std: float, segment: Optional[Dict] = None) -> float:
        #calculate the segment score
        score = (mean_pitch - self.pitch_threshold) / self.pitch_threshold
        #calculate the standard deviation adjustment
        std_adjustment = (pitch_std / 100.0) * self.pitch_std_weight
        #calculate the spectral contribution
        spectral_contribution = 0.0
        #if the segment is not empty and the spectral weight is greater than 0, then:
        if segment and self.spectral_weight > 0:
            #calculate the spectral contribution
            spectral_contribution = self.compute_spectral_score(segment)
        #return the score plus the standard deviation adjustment plus the spectral contribution
        return score + std_adjustment + spectral_contribution
#this definition is basically used to compute the overall feature score for the overall classification
    def compute_overall_feature_score(self, average_frequency: float, peak_frequency: float,
                                     frequency_std: float, features: Optional[Dict] = None) -> float:
        #initialize the score to 0
        score = 0.0
        #calculate the average frequency score
        avg_freq_score = (average_frequency - self.pitch_threshold) / self.pitch_threshold
        score += avg_freq_score * self.avg_freq_weight
        #calculate the peak frequency score
        peak_freq_score = (peak_frequency - 300.0) / 300.0
        score += peak_freq_score * self.peak_freq_weight
        #calculate the frequency standard deviation score
        freq_std_score = (frequency_std / 100.0) * 0.1
        score += freq_std_score * self.freq_std_weight
        #if the features are not empty and the spectral weight is greater than 0, then:
        if features and self.spectral_weight > 0:
            #get the spectral centroid and zero crossing rate from the features
            spectral_centroid = features.get('spectral_centroid', 0.0)
            zcr = features.get('zero_crossing_rate', 0.0)
            #calculate the spectral centroid score
            if spectral_centroid > 0:#if the spectral centroid is greater than 0, then:
                #calculate the spectral centroid score
                centroid_score = (spectral_centroid - 2000.0) / 2000.0
                score += centroid_score * self.spectral_weight * 0.3
            #calculate the zero crossing rate score
            if zcr > 0:#if the zero crossing rate is greater than 0, then:
                #calculate the zero crossing rate score
                zcr_score = -(zcr - 0.05) / 0.05
                score += zcr_score * self.spectral_weight * 0.2
        #return the overall feature score
        return score
#classify the gender of the speaker based on the features
    def classify(self, features: Dict) -> Dict:
        #if the segments are not in the features, then:
        if 'segments' not in features:
            raise ValueError("Features must contain 'segments' list for segment-based classification")
        #get the segments from the features and check if it is empty
        segments = features['segments']
        if not segments:
            raise ValueError("No segments found in features")
        #initialize the lists to store the segment predictions, scores, and pitches
        segment_predictions = []
        segment_scores = []
        all_pitches = []
        #iterate through the segments and compute the segment score, prediction, and pitch
        for segment in segments:
            mean_pitch = segment['mean_pitch']
            pitch_std = segment['pitch_std']
            all_pitches.append(mean_pitch)
            #compute the segment score for the segment based classification
            score = self.compute_segment_score(mean_pitch, pitch_std, segment)
            segment_scores.append(score)
            segment_predictions.append('female' if score >= 0 else 'male')
        #calculate the average pitch from the segments
        avg_pitch = np.mean(all_pitches)
        #calculate the number of male and female votes
        male_votes = sum(1 for pred in segment_predictions if pred == 'male')
        female_votes = len(segment_predictions) - male_votes
        #initialize the overall feature score to 0
        overall_feature_score = 0.0
        #check if the overall features are not empty
        has_overall_features = all(key in features for key in ['average_frequency', 'peak_frequency', 'frequency_std'])
        if has_overall_features:#if the overall features are not empty, then:
            #compute the overall feature score for the overall classification
            overall_feature_score = self.compute_overall_feature_score(
                features['average_frequency'],
                features['peak_frequency'],
                features['frequency_std'],
                features
            )
        #calculate the segment margin with absolute value and divide by the number of segments
        segment_margin = abs(male_votes - female_votes) / len(segment_predictions) if segment_predictions else 0
        #if the male votes are greater than the female votes, then (this is where the majority votes is decided):
        if male_votes > female_votes:
            #set the base prediction to male
            base_prediction = 'male'
        elif female_votes > male_votes:
            #set the base prediction to female
            base_prediction = 'female'
        else:
            #if the average pitch is less than the pitch threshold, then:   
            if avg_pitch < self.pitch_threshold:
                #set the base prediction to male
                base_prediction = 'male'
            #if the average pitch is greater than or equal to the female threshold, then:
            elif avg_pitch >= self.female_threshold:
                #set the base prediction to female
                base_prediction = 'female'
            else:
                #calculate the zone midpoint 
                zone_midpoint = (self.pitch_threshold + self.female_threshold) / 2
                base_prediction = 'female' if avg_pitch >= zone_midpoint else 'male'
        #set the final prediction to the base prediction
        final_prediction = base_prediction
        overall_feature_override = False
        if has_overall_features and segment_margin < 0.3:
            #if the overall feature score is greater than 0.1 and the base prediction is male, then:
            if overall_feature_score > 0.1 and base_prediction == 'male':
                #set the final prediction to female
                final_prediction = 'female'
                #set the overall feature override to true
                overall_feature_override = True
            elif overall_feature_score < -0.1 and base_prediction == 'female':
                #set the final prediction to male
                final_prediction = 'male'
                #set the overall feature override to true
                overall_feature_override = True
        #return everything needed for the result
        return {
            'prediction': final_prediction,
            'base_prediction': base_prediction,
            'segment_predictions': segment_predictions,
            'segment_scores': segment_scores,
            'male_votes': male_votes,
            'female_votes': female_votes,
            'confidence': max(male_votes, female_votes) / len(segment_predictions),
            'num_segments': len(segments),
            'average_pitch': avg_pitch,
            'overall_feature_score': overall_feature_score if has_overall_features else None,
            'overall_feature_override': overall_feature_override,
            'segment_margin': segment_margin,
            'average_frequency': features.get('average_frequency'),
            'peak_frequency': features.get('peak_frequency'),
            'frequency_std': features.get('frequency_std')
        }
#this definition is basically used to classify a bunch of features at once
    def classify_batch(self, features_list: List[Dict]) -> List[Dict]:
        return [self.classify(features) for features in features_list]
#this definition is used to classify a single feature
class GenderClassifier:
    #initialize the classifier with the default values (adaptive threshold is also used)
    def __init__(self, pitch_threshold: float = DEFAULT_PITCH_THRESHOLD,
                 female_threshold: float = DEFAULT_FEMALE_THRESHOLD,
                 mean_pitch_weight: float = DEFAULT_MEAN_PITCH_WEIGHT,
                 pitch_std_weight: float = DEFAULT_PITCH_STD_WEIGHT,
                 avg_freq_weight: float = DEFAULT_AVG_FREQ_WEIGHT,
                 peak_freq_weight: float = DEFAULT_PEAK_FREQ_WEIGHT,
                 freq_std_weight: float = DEFAULT_FREQ_STD_WEIGHT,
                 spectral_weight: float = DEFAULT_SPECTRAL_WEIGHT,
                 use_adaptive_threshold: bool = USE_ADAPTIVE_THRESHOLD,
                 adaptive_alpha: float = ADAPTIVE_THRESHOLD_ALPHA):
        #initialize the classifier with the default values
        self.classifier = CustomRuleBasedClassifier(
            pitch_threshold=pitch_threshold,
            female_threshold=female_threshold,
            mean_pitch_weight=mean_pitch_weight,
            pitch_std_weight=pitch_std_weight,
            avg_freq_weight=avg_freq_weight,
            peak_freq_weight=peak_freq_weight,
            freq_std_weight=freq_std_weight,
            spectral_weight=spectral_weight,
            use_adaptive_threshold=use_adaptive_threshold,
            adaptive_alpha=adaptive_alpha
        )
#training the classifier from the training data
    def calibrate_from_training(self, training_features: List[Dict]):
        
        self.classifier.calibrate_thresholds_from_training(training_features)
#classify the gender of the speaker based on the features...return everything needed for the result
    def classify(self, features: Dict) -> Dict:
        result = self.classifier.classify(features)
        result['final_prediction'] = result['prediction']
        result['method'] = 'custom_rule_based'
        return result