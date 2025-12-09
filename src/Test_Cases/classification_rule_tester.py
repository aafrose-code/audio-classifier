import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Code_Logic'))
from src.Code_Logic.extract_voice_metadata import VoiceFeatureExtractor
#test the classification rule tester (this is the only test file that doesn't use mock objects and assertions because it is a standalone file)
class ClassificationRuleTester:
    #initialize the tester
    def __init__(self):
        self.extractor = VoiceFeatureExtractor()
        self.test_results = []
    #test the rule
    def test_rule(self,
                  audio_path: str,
                  true_label: str,
                  avg_freq_threshold: float = 165.0,
                  peak_freq_threshold: float = 300.0,
                  std_threshold: float = None) -> Dict:
        #features extraction using the voice feature extractor
        features = self.extractor.extract_features(audio_path, use_segments=False)
        #get the average frequency, peak frequency, and frequency standard deviation from the features
        avg_freq = features['average_frequency']
        peak_freq = features['peak_frequency']
        freq_std = features['frequency_std']
        #initialize the prediction and reasoning (they will be used to store the prediction and the reasoning for the prediction)
        prediction = None
        reasoning = []
        #if the average frequency is less than the average frequency threshold, then:
        if avg_freq < avg_freq_threshold:
            prediction = 'male'#set the prediction to male
            reasoning.append(f"Average frequency ({avg_freq:.2f} Hz) < threshold ({avg_freq_threshold} Hz)")
        elif avg_freq >= avg_freq_threshold:#if the average frequency is greater than or equal to the average frequency threshold, then:
            prediction = 'female'#set the prediction to female
            reasoning.append(f"Average frequency ({avg_freq:.2f} Hz) >= threshold ({avg_freq_threshold} Hz)")
        #if the peak frequency is less than the peak frequency threshold, then:
        if peak_freq < peak_freq_threshold:
            peak_suggests = 'male'#set the peak suggests to male
            reasoning.append(f"Peak frequency ({peak_freq:.2f} Hz) < threshold ({peak_freq_threshold} Hz) suggests {peak_suggests}")#add the reasoning to the reasoning list
        else:
            #if the peak frequency is greater than or equal to the peak frequency threshold, then:
            peak_suggests = 'female'#set the peak suggests to female
            reasoning.append(f"Peak frequency ({peak_freq:.2f} Hz) >= threshold ({peak_freq_threshold} Hz) suggests {peak_suggests}")#add the reasoning to the reasoning list

        if std_threshold:#if the frequency standard deviation threshold is not None, then:
            if freq_std < std_threshold:#if the frequency standard deviation is less than the frequency standard deviation threshold, then:
                std_suggests = 'male'#set the std suggests to male
            else:#if the frequency standard deviation is greater than or equal to the frequency standard deviation threshold, then:
                std_suggests = 'female'#set the std suggests to female
            reasoning.append(f"Frequency std ({freq_std:.2f} Hz) suggests {std_suggests}")#add the reasoning to the reasoning list
        #check if the prediction is correct
        correct = prediction.lower() == true_label.lower()
        #create the result dictionary
        result = {
            'audio_path': audio_path,
            'true_label': true_label,
            'prediction': prediction,
            'correct': correct,
            'average_frequency': avg_freq,
            'peak_frequency': peak_freq,
            'frequency_std': freq_std,
            'reasoning': reasoning,
            'thresholds': {
                'avg_freq': avg_freq_threshold,
                'peak_freq': peak_freq_threshold,
                'std': std_threshold
            }
        }
        #add the result to the test results list
        self.test_results.append(result)
        return result
    #test the batch
    def test_batch(self,
                   audio_paths: List[str],
                   labels: List[str],
                   avg_freq_threshold: float = 165.0,
                   peak_freq_threshold: float = 300.0,
                   std_threshold: float = None) -> List[Dict]:

        results = []#initialize the results list
        for audio_path, label in zip(audio_paths, labels):#test the rule for each audio path and label by looping through the audio paths and labels
            result = self.test_rule(audio_path, label, avg_freq_threshold,
                                   peak_freq_threshold, std_threshold)
            results.append(result)#add the result to the results list
        return results#return the results
    #evaluate the rules
    def evaluate_rules(self,
                       avg_freq_threshold: float = 165.0,
                       peak_freq_threshold: float = 300.0,
                       std_threshold: float = None) -> Dict:

        if not self.test_results:#if the test results list is empty, then:
            #return an empty dictionary
            return {}
        #get the number of correct predictions
        correct = sum(1 for r in self.test_results if r['correct'])
        total = len(self.test_results)#get the total number of predictions
        accuracy = correct / total if total > 0 else 0.0#calculate the accuracy
        male_correct = sum(1 for r in self.test_results if r['true_label'].lower() == 'male' and r['correct'])#get the number of correct male predictions
        male_total = sum(1 for r in self.test_results if r['true_label'].lower() == 'male')#get the total number of male predictions
        male_accuracy = male_correct / male_total if male_total > 0 else 0.0#calculate the male accuracy
#get the number of correct female predictions   
        female_correct = sum(1 for r in self.test_results if r['true_label'].lower() == 'female' and r['correct'])#get the number of correct female predictions
        female_total = sum(1 for r in self.test_results if r['true_label'].lower() == 'female')#get the total number of female predictions
        female_accuracy = female_correct / female_total if female_total > 0 else 0.0#calculate the female accuracy
        #return the evaluation dictionary
        return {
            'overall_accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'male_accuracy': male_accuracy,
            'male_samples': male_total,
            'female_accuracy': female_accuracy,
            'female_samples': female_total,
            'thresholds': {
                'avg_freq': avg_freq_threshold,
                'peak_freq': peak_freq_threshold,
                'std': std_threshold
            }
        }
    #print the evaluation
    def print_evaluation(self):
        evaluation = self.evaluate_rules()#evaluate the rules

        if not evaluation:#if the evaluation dictionary is empty, then:
            #print a message
            print("No test results available. Run test_rule() or test_batch() first.")
            return
        #print the evaluation   
        print("\nCLASSIFICATION RULE EVALUATION")
        #print the overall accuracy
        print(f"\nOverall Accuracy: {evaluation['overall_accuracy']:.2%}")
        print(f"  Correct: {evaluation['correct_predictions']}/{evaluation['total_samples']}")
        #print the male accuracy
        print(f"\nMale Accuracy: {evaluation['male_accuracy']:.2%}")
        print(f"  Samples: {evaluation['male_samples']}")
        #print the female accuracy
        print(f"\nFemale Accuracy: {evaluation['female_accuracy']:.2%}")
        print(f"  Samples: {evaluation['female_samples']}")
        #print the thresholds used
        print(f"\nThresholds Used:")
        print(f"  Average Frequency: {evaluation['thresholds']['avg_freq']} Hz")
        print(f"  Peak Frequency: {evaluation['thresholds']['peak_freq']} Hz")
        if evaluation['thresholds']['std']:#if the frequency standard deviation threshold is not None, then:
            #print the frequency standard deviation threshold
            print(f"  Frequency Std Dev: {evaluation['thresholds']['std']} Hz")

    #find the optimal thresholds
    def find_optimal_thresholds(self,
                               audio_paths: List[str],
                               labels: List[str],
                               avg_freq_range: Tuple[float, float] = (140.0, 200.0),
                               peak_freq_range: Tuple[float, float] = (250.0, 400.0),
                               step: float = 5.0) -> Dict:
        #initialize the best accuracy, best thresholds, and best results
        best_accuracy = 0.0
        best_thresholds = {}
        best_results = []
        #get the start and end of the average frequency range
        avg_start, avg_end = avg_freq_range
        #get the start and end of the peak frequency range
        peak_start, peak_end = peak_freq_range
        for avg_thresh in np.arange(avg_start, avg_end + step, step):#loop through the average frequency range
            for peak_thresh in np.arange(peak_start, peak_end + step, step):#loop through the peak frequency range
                #initialize the test results
                self.test_results = []
                self.test_batch(audio_paths, labels, avg_thresh, peak_thresh)
                #evaluate the rules
                eval_result = self.evaluate_rules(avg_thresh, peak_thresh)
                #if the overall accuracy is greater than the best accuracy, then:
                if eval_result['overall_accuracy'] > best_accuracy:
                    #set the best accuracy to the overall accuracy
                    best_accuracy = eval_result['overall_accuracy']
                    #set the best thresholds to the average and peak thresholds
                    best_thresholds = {
                        'avg_freq': avg_thresh,
                        'peak_freq': peak_thresh
                    }
                    #set the best results to the test results (copy of the test results)
                    best_results = self.test_results.copy()
        #set the test results to the best results
        self.test_results = best_results
        #return the best accuracy and thresholds
        return {
            'best_accuracy': best_accuracy,
            'best_thresholds': best_thresholds
        }
    #get the incorrect predictions
    def get_incorrect_predictions(self) -> List[Dict]:
        return [r for r in self.test_results if not r['correct']]#return the incorrect predictions by filtering the test results for incorrect predictions
    #print the incorrect predictions
    def print_incorrect_predictions(self):
        incorrect = self.get_incorrect_predictions()#get the incorrect predictions
        #if the incorrect predictions list is empty, then:
        if not incorrect:#if the incorrect predictions list is empty, then:
            #print a message
            print("\nAll predictions were correct!")
            return
        #print the number of incorrect predictions
        print(f"\nIncorrect Predictions ({len(incorrect)}):")
        #print the incorrect predictions by looping through the incorrect predictions and printing the results    
        for result in incorrect:#loop through the incorrect predictions and print the results
            print(f"\nFile: {result['audio_path']}")#print the audio path
            print(f"  True Label: {result['true_label']}")#print the true label
            print(f"  Prediction: {result['prediction']}")#print the prediction
            print(f"  Average Frequency: {result['average_frequency']:.2f} Hz")#print the average frequency
            print(f"  Peak Frequency: {result['peak_frequency']:.2f} Hz")#print the peak frequency
            print(f"  Frequency Std: {result['frequency_std']:.2f} Hz")#print the frequency standard deviation
            print(f"  Reasoning: {'; '.join(result['reasoning'])}")#print the reasoning