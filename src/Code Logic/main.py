import os
import json
import csv
import numpy as np
from typing import List, Dict, Tuple
from extract_voice_metadata import VoiceFeatureExtractor
from classification import GenderClassifier, CustomRuleBasedClassifier


class GenderIdentificationSystem:
    
    def __init__(self, pitch_threshold: float = 165.0,
                 mean_pitch_weight: float = 0.7,
                 pitch_std_weight: float = 0.3):
        self.feature_extractor = VoiceFeatureExtractor()
        self.classifier = GenderClassifier(
            pitch_threshold=pitch_threshold,
            mean_pitch_weight=mean_pitch_weight,
            pitch_std_weight=pitch_std_weight
        )
        self.training_data = []
        self.test_data = []
    
    def collect_features(self, audio_paths: List[str], labels: List[str] = None) -> List[Dict]:
        print(f"\nExtracting segment-based features from {len(audio_paths)} audio files...")
        features_list = self.feature_extractor.extract_features_batch(audio_paths, use_segments=True)
        
        if labels:
            if len(labels) != len(features_list):
                print("Warning: Number of labels doesn't match number of successfully extracted features")
            else:
                for i, features in enumerate(features_list):
                    features['true_label'] = labels[i]
        
        return features_list
    
    def analyze_training_data(self, training_audio_paths: List[str], training_labels: List[str]):
        print("\n" + "="*60)
        print("ANALYZING TRAINING DATA")
        print("="*60)
        
        training_features = self.collect_features(training_audio_paths, training_labels)
        self.training_data = training_features
        
        male_pitches = []
        female_pitches = []
        male_stds = []
        female_stds = []
        
        for features in training_features:
            if 'true_label' in features:
                label = features['true_label'].lower()
                for segment in features.get('segments', []):
                    if label == 'male':
                        male_pitches.append(segment['mean_pitch'])
                        male_stds.append(segment['pitch_std'])
                    elif label == 'female':
                        female_pitches.append(segment['mean_pitch'])
                        female_stds.append(segment['pitch_std'])
        
        if male_pitches and female_pitches:
            print(f"\nTraining Data Analysis:")
            print(f"  Male segments: {len(male_pitches)}")
            print(f"    Mean pitch: {np.mean(male_pitches):.2f} Hz (std: {np.std(male_pitches):.2f})")
            print(f"    Mean pitch std dev: {np.mean(male_stds):.2f} Hz")
            print(f"  Female segments: {len(female_pitches)}")
            print(f"    Mean pitch: {np.mean(female_pitches):.2f} Hz (std: {np.std(female_pitches):.2f})")
            print(f"    Mean pitch std dev: {np.mean(female_stds):.2f} Hz")
        
        print(f"\nAnalysis complete! Analyzed {len(training_features)} samples.")
    
    def test_samples(self, test_audio_paths: List[str], test_labels: List[str] = None) -> List[Dict]:
        print("\n" + "="*60)
        print("TESTING PHASE")
        print("="*60)
        
        test_features = self.collect_features(test_audio_paths, test_labels)
        self.test_data = test_features
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        print(f"\nClassifying {len(test_features)} test samples...")
        print("-" * 60)
        
        for features in test_features:
            features_only = {k: v for k, v in features.items() 
                           if k not in ['audio_path', 'true_label']}
            
            prediction_result = self.classifier.classify(features_only)
            
            result = {
                'audio_path': features.get('audio_path', 'unknown'),
                'final_prediction': prediction_result['final_prediction'],
                'method': prediction_result['method'],
                'num_segments': prediction_result['num_segments'],
                'male_votes': prediction_result['male_votes'],
                'female_votes': prediction_result['female_votes'],
                'confidence': prediction_result['confidence']
            }
            
            if 'segment_predictions' in prediction_result:
                result['segment_predictions'] = prediction_result['segment_predictions']
                result['segment_scores'] = prediction_result['segment_scores']
            
            if 'true_label' in features:
                result['true_label'] = features['true_label']
                result['correct'] = (prediction_result['final_prediction'].lower() == 
                                   features['true_label'].lower())
                
                if result['correct']:
                    correct_predictions += 1
                total_predictions += 1
                
                status = "✓" if result['correct'] else "✗"
                print(f"{status} {os.path.basename(result['audio_path'])}: "
                      f"Predicted={result['final_prediction']}, "
                      f"True={result['true_label']}, "
                      f"Segments={result['num_segments']}, "
                      f"Votes: M={result['male_votes']}/F={result['female_votes']}, "
                      f"Confidence={result['confidence']:.2%}")
            else:
                print(f"? {os.path.basename(result['audio_path'])}: "
                      f"Predicted={result['final_prediction']}, "
                      f"Segments={result['num_segments']}, "
                      f"Votes: M={result['male_votes']}/F={result['female_votes']}, "
                      f"Confidence={result['confidence']:.2%}")
            
            results.append(result)
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print("-" * 60)
            print(f"\nTest Results Summary:")
            print(f"Total Samples: {total_predictions}")
            print(f"Correct Predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.2%}")
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str = "results.json"):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def save_results_csv(self, results: List[Dict], output_path: str = "results.csv"):
        if not results:
            return
        
        fieldnames = list(results[0].keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to {output_path}")
    
    def get_classifier_info(self) -> Dict:
        return {
            'method': 'custom_rule_based',
            'pitch_threshold': self.classifier.classifier.pitch_threshold,
            'mean_pitch_weight': self.classifier.classifier.mean_pitch_weight,
            'pitch_std_weight': self.classifier.classifier.pitch_std_weight
        }


def find_audio_files(directory: str, extensions: tuple = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')) -> List[str]:
    audio_files = []
    if not os.path.exists(directory):
        return audio_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(root, file))
    
    return sorted(audio_files)


def load_training_data():
    training_dir = "audio_samples/training"
    
    if os.path.exists(training_dir):
        male_dir = os.path.join(training_dir, "male")
        female_dir = os.path.join(training_dir, "female")
        
        audio_paths = []
        labels = []
        
        if os.path.exists(male_dir):
            male_files = find_audio_files(male_dir)
            audio_paths.extend(male_files)
            labels.extend(['male'] * len(male_files))
        
        if os.path.exists(female_dir):
            female_files = find_audio_files(female_dir)
            audio_paths.extend(female_files)
            labels.extend(['female'] * len(female_files))
        
        if audio_paths:
            return audio_paths, labels
    
    if os.path.exists(training_dir):
        all_files = find_audio_files(training_dir)
        audio_paths = []
        labels = []
        
        for file_path in all_files:
            filename = os.path.basename(file_path).lower()
            if 'male' in filename or 'm_' in filename:
                audio_paths.append(file_path)
                labels.append('male')
            elif 'female' in filename or 'f_' in filename or 'woman' in filename:
                audio_paths.append(file_path)
                labels.append('female')
        
        if audio_paths:
            return audio_paths, labels
    
    training_audio_paths = [
    ]
    
    training_labels = [
    ]
    
    if training_audio_paths and len(training_audio_paths) == len(training_labels):
        return training_audio_paths, training_labels
    
    return None, None


def load_test_data():
    test_dir = "audio_samples/test"
    
    if os.path.exists(test_dir):
        male_dir = os.path.join(test_dir, "male")
        female_dir = os.path.join(test_dir, "female")
        
        audio_paths = []
        labels = []
        
        if os.path.exists(male_dir):
            male_files = find_audio_files(male_dir)
            audio_paths.extend(male_files)
            labels.extend(['male'] * len(male_files))
        
        if os.path.exists(female_dir):
            female_files = find_audio_files(female_dir)
            audio_paths.extend(female_files)
            labels.extend(['female'] * len(female_files))
        
        if audio_paths:
            return audio_paths, labels
        
        all_files = find_audio_files(test_dir)
        if all_files:
            audio_paths = []
            labels = []
            
            for file_path in all_files:
                filename = os.path.basename(file_path).lower()
                audio_paths.append(file_path)
                
                if 'male' in filename or 'm_' in filename:
                    labels.append('male')
                elif 'female' in filename or 'f_' in filename or 'woman' in filename:
                    labels.append('female')
                else:
                    labels.append(None)
            
            return audio_paths, labels if any(labels) else None
    
    test_audio_paths = [
    ]
    
    test_labels = [
    ]
    
    if test_audio_paths:
        return test_audio_paths, test_labels if test_labels else None
    
    return None, None


def main():
    print("="*60)
    print("Gender Identification Through Audio Analysis")
    print("="*60)
    
    system = GenderIdentificationSystem(
        pitch_threshold=165.0,
        mean_pitch_weight=0.7,
        pitch_std_weight=0.3
    )
    
    classifier_info = system.get_classifier_info()
    print(f"\nClassifier Configuration:")
    print(f"  Method: {classifier_info['method']}")
    print(f"  Pitch Threshold: {classifier_info['pitch_threshold']} Hz")
    print(f"  Mean Pitch Weight: {classifier_info['mean_pitch_weight']:.2f}")
    print(f"  Pitch Std Dev Weight: {classifier_info['pitch_std_weight']:.2f}")
    
    print("\n" + "="*60)
    print("LOADING TRAINING DATA (Optional - for analysis)")
    print("="*60)
    
    training_audio_paths, training_labels = load_training_data()
    
    if training_audio_paths and training_labels:
        print(f"\nFound {len(training_audio_paths)} training samples:")
        print(f"  Male: {sum(1 for l in training_labels if l == 'male')}")
        print(f"  Female: {sum(1 for l in training_labels if l == 'female')}")
        
        try:
            system.analyze_training_data(training_audio_paths, training_labels)
        except Exception as e:
            print(f"\nError during analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo training data found (optional).")
        print("The rule-based classifier will work without training data.")
        print("\nTo analyze training data, organize files in:")
        print("  audio_samples/training/male/")
        print("  audio_samples/training/female/")
    
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    test_audio_paths, test_labels = load_test_data()
    
    if not test_audio_paths:
        print("\nNo test data found!")
        print("\nTo test the system, place test audio files in:")
        print("  audio_samples/test/")
        print("\nOR update the hardcoded test paths in main.py")
        return
    
    print(f"\nFound {len(test_audio_paths)} test samples")
    if test_labels:
        print(f"  With labels: {sum(1 for l in test_labels if l is not None)}")
    
    try:
        results = system.test_samples(test_audio_paths, test_labels)
        
        output_dir = "results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        system.save_results(results, os.path.join(output_dir, "results.json"))
        system.save_results_csv(results, os.path.join(output_dir, "results.csv"))
        
        print("\n" + "="*60)
        print("PROCESS COMPLETE")
        print("="*60)
        print(f"\nResults saved to '{output_dir}/' directory")
        print("\nClassifier uses custom rule-based method with:")
        print("  - 1-second segment analysis")
        print("  - Weighted scoring (mean pitch + pitch std dev)")
        print("  - Majority voting across segments")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
