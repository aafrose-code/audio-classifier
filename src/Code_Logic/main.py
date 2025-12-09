import os
import numpy as np#need numpy for the mathematical operations (basically a math library...doesnt have a direct math library like c does)
from typing import List, Dict, Tuple
from extract_voice_metadata import VoiceFeatureExtractor
from classification import GenderClassifier
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data_Logic'))
from Data_Logic.utils import save_to_csv
#load the default values for the classifier
from config import (LOGGING_DIR, DEFAULT_PITCH_THRESHOLD, DEFAULT_FEMALE_THRESHOLD,
                   DEFAULT_MEAN_PITCH_WEIGHT, DEFAULT_PITCH_STD_WEIGHT, DEFAULT_AVG_FREQ_WEIGHT,
                   DEFAULT_PEAK_FREQ_WEIGHT, DEFAULT_FREQ_STD_WEIGHT, DEFAULT_SPECTRAL_WEIGHT,
                   USE_ADAPTIVE_THRESHOLD, ADAPTIVE_THRESHOLD_ALPHA)
from path_helpers import get_project_root, get_voice_samples_dir, get_audio_samples_dir, ensure_directory_exists
from audio_helpers import find_audio_files, infer_gender_from_filename

#this is the main class for the gender identification system
class GenderIdentificationSystem:
#initialize the system with the default values for the classifier
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
                 #initialize the feature extractor and the classifier
        self.feature_extractor = VoiceFeatureExtractor()
        #initialize the classifier with the default values for the classifier
        self.classifier = GenderClassifier(
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
        #initialize the training and test data as empty lists
        self.training_data = []
        self.test_data = []
#collect the features from the audio files
    def collect_features(self, audio_paths: List[str], labels: List[str] = None) -> List[Dict]:#this is used to collect the features from the audio files
        print(f"\nExtracting segment-based features from {len(audio_paths)} audio files...")
        features_list = self.feature_extractor.extract_features_batch(audio_paths, use_segments=True)
        print()#print the number of audio files being processed
        #if the labels are provided, then:
        if labels:
            if len(labels) != len(features_list):#if the number of labels does not match the number of successfully extracted features, then:
                print("Warning: Number of labels doesn't match number of successfully extracted features")#print a warning message
            else:
                for i, features in enumerate(features_list):#enumerate the features and add the labels to the features
                    features['true_label'] = labels[i]#add the label to the features

        return features_list#return the features from the audio files
#analyze the training data
    def analyze_training_data(self, training_audio_paths: List[str], training_labels: List[str]):
        print("\nANALYZING TRAINING DATA")
#collect the features from the training audio files
        training_features = self.collect_features(training_audio_paths, training_labels)
        self.training_data = training_features #store the training features in the training data array
#initialize the lists to store the pitches for the male and female training data
        male_pitches = []
        female_pitches = []
        male_stds = []
        female_stds = []
#iterate through the training features and store the pitches for the male and female training data
        for features in training_features:
            if 'true_label' in features:#if the true label is in the features, then:
                label = features['true_label'].lower()#get the label from the features
                for segment in features.get('segments', []):#iterate through the segments in the training data and store the pitches for the male and female training data
                    if label == 'male':#if the label is male, then:
                        male_pitches.append(segment['mean_pitch'])#store the mean pitch for the male training data
                        male_stds.append(segment['pitch_std'])#store the standard deviation of the pitch for the male training data
                    elif label == 'female':#if the label is female, then:
                        female_pitches.append(segment['mean_pitch'])#store the mean pitch for the female training data
                        female_stds.append(segment['pitch_std'])#store the standard deviation of the pitch for the female training data

        if male_pitches and female_pitches:#if the male and female pitches are not empty, then:
            #print the training data analysis
            print(f"\nTraining Data Analysis:")
            print(f"  Male segments: {len(male_pitches)}")
            print(f"    Mean pitch: {np.mean(male_pitches):.2f} Hz (std: {np.std(male_pitches):.2f})")
            print(f"    Mean pitch std dev: {np.mean(male_stds):.2f} Hz")
            print(f"  Female segments: {len(female_pitches)}")
            print(f"    Mean pitch: {np.mean(female_pitches):.2f} Hz (std: {np.std(female_pitches):.2f})")
            print(f"    Mean pitch std dev: {np.mean(female_stds):.2f} Hz")

        if USE_ADAPTIVE_THRESHOLD:#if the adaptive threshold is used, then:
            #tell the classifier from the training data to calibrate the thresholds
            self.classifier.calibrate_from_training(training_features)
#print the training data analysis complete message
        print(f"\nAnalysis complete! Analyzed {len(training_features)} samples.")
#test the samples
    def test_samples(self, test_audio_paths: List[str], test_labels: List[str] = None) -> List[Dict]:
        print("\nTESTING PHASE")
#collect the features from the test audio files
        test_features = self.collect_features(test_audio_paths, test_labels)
        self.test_data = test_features
        #initialize the list to store the results
        results = []
        correct_predictions = 0
        total_predictions = 0
        #print the number of test samples being classified
        print(f"\nClassifying {len(test_features)} test samples...")
        #iterate through the test features and classify the samples
        for features in test_features:
            features_only = {k: v for k, v in features.items()#remove the audio path and true label from the features
                           if k not in ['audio_path', 'true_label']}
#classify the samples
            prediction_result = self.classifier.classify(features_only)
#store the results in a dictionary
            result = {
                'audio_path': features.get('audio_path', 'unknown'),
                'final_prediction': prediction_result['final_prediction'],
                'method': prediction_result['method'],
                'num_segments': prediction_result['num_segments'],
                'male_votes': prediction_result['male_votes'],
                'female_votes': prediction_result['female_votes'],
                'confidence': prediction_result['confidence']
            }
#if the segment predictions are in the prediction result, then:
            if 'segment_predictions' in prediction_result:
                result['segment_predictions'] = prediction_result['segment_predictions']#store the segment predictions in the result
                result['segment_scores'] = prediction_result['segment_scores']#store the segment scores in the result
#if the overall feature override is in the prediction result, then:
            overall_override = prediction_result.get('overall_feature_override', False)#get the overall feature override from the prediction result
            overall_score = prediction_result.get('overall_feature_score')#get the overall feature score from the prediction result
            segment_margin = prediction_result.get('segment_margin', 0)#get the segment margin from the prediction result
            base_prediction = prediction_result.get('base_prediction', prediction_result['final_prediction'])#get the base prediction from the prediction result
#
            if 'true_label' in features:#if the true label is in the features, then:
                result['true_label'] = features['true_label']#replace the true label in the result with the true label from the features
                result['correct'] = (prediction_result['final_prediction'].lower() ==#check if the final prediction is the same as the true label
                                   features['true_label'].lower())
                if result['correct']:#if the final prediction is the same as the true label, then:
                    correct_predictions += 1#increment the correct predictions
                total_predictions += 1#increment the total predictions
                status = "CORRECT" if result['correct'] else "INCORRECT"#set the status to CORRECT if the final prediction is the same as the true label, otherwise set it to INCORRECT
                prediction_line = (f"{status} {os.path.basename(result['audio_path'])}: "#print the audio path and the status
                                 f"Predicted={result['final_prediction']}, "
                                 f"True={result['true_label']}, "
                                 f"Segments={result['num_segments']}, "
                                 f"Votes: M={result['male_votes']}/F={result['female_votes']}, "
                                 f"Confidence={result['confidence']:.2%}")
                #if the overall feature override is in the prediction result, then:
                if overall_override:
                    #print the overall feature override activated message
                    prediction_line += f"\n  OVERALL FEATURE OVERRIDE ACTIVATED (Confidence close to 50%: {result['confidence']:.2%})"
                    prediction_line += f"\n     Segments voted: {base_prediction.upper()}, "
                    prediction_line += f"Overall features suggest: {result['final_prediction'].upper()}"
                    prediction_line += f"\n     Overall Feature Score: {overall_score:.3f}, "
                    prediction_line += f"Segment Margin: {segment_margin:.1%}"
                elif overall_score is not None and segment_margin < 0.3:#if the overall score is not None and the segment margin is less than 0.3, then:
                    #print the overall features available message
                    prediction_line += f"\n  Overall features available (Score: {overall_score:.3f}) but not strong enough to override"#print the overall features available message
                #print the prediction line
                print(prediction_line)
            else:#if the true label is not in the features, then:
                overall_override = prediction_result.get('overall_feature_override', False)#get the overall feature override from the prediction result
                overall_score = prediction_result.get('overall_feature_score')#get the overall feature score from the prediction result
                segment_margin = prediction_result.get('segment_margin', 0)#get the segment margin from the prediction result
                base_prediction = prediction_result.get('base_prediction', prediction_result['final_prediction'])#get the base prediction from the prediction result

                prediction_line = (f"UNKNOWN {os.path.basename(result['audio_path'])}: "#print the audio path and the unknown message
                                  f"Predicted={result['final_prediction']}, "#print the final prediction
                                  f"Segments={result['num_segments']}, "
                                  f"Votes: M={result['male_votes']}/F={result['female_votes']}, "#print the votes
                                  f"Confidence={result['confidence']:.2%}")#print the confidence

                if overall_override:#if the overall feature override is in the prediction result, then:
                    prediction_line += f"\n  OVERALL FEATURE OVERRIDE ACTIVATED (Confidence close to 50%: {result['confidence']:.2%})"#print the overall feature override activated message
                    prediction_line += f"\n     Segments voted: {base_prediction.upper()}, "#print the segments voted
                    prediction_line += f"Overall features suggest: {result['final_prediction'].upper()}"#print the overall features suggest
                    prediction_line += f"\n     Overall Feature Score: {overall_score:.3f}, "#print the overall feature score
                    prediction_line += f"Segment Margin: {segment_margin:.1%}"#print the segment margin
                elif overall_score is not None and segment_margin < 0.3:#if the overall score is not None and the segment margin is less than 0.3, then:
                    prediction_line += f"\n  Overall features available (Score: {overall_score:.3f}) but not strong enough to override"#print the overall features available message
                #print the prediction line
                print(prediction_line)
            #append the result to the results list
            results.append(result)
        #print the test results summary
        if total_predictions > 0:#if the total predictions is greater than 0, then:
            accuracy = correct_predictions / total_predictions#calculate the accuracy
            print(f"\nTest Results Summary:")#print the test results summary
            print(f"Total Samples: {total_predictions}")#print the total samples
            print(f"Correct Predictions: {correct_predictions}")#print the correct predictions
            print(f"Accuracy: {accuracy:.2%}")#print the accuracy
        #return the results
        return results

#save the results to a csv file

    def save_results_csv(self, results: List[Dict], output_path: str):
        save_to_csv(results, output_path)#save the results to a csv file

    def get_classifier_info(self) -> Dict:#this is used to get the classifier information
        return {
            'method': 'custom_rule_based',
            'pitch_threshold': self.classifier.classifier.pitch_threshold,
            'mean_pitch_weight': self.classifier.classifier.mean_pitch_weight,
            'pitch_std_weight': self.classifier.classifier.pitch_std_weight,
            'avg_freq_weight': self.classifier.classifier.avg_freq_weight,
            'peak_freq_weight': self.classifier.classifier.peak_freq_weight,
            'freq_std_weight': self.classifier.classifier.freq_std_weight,
            'spectral_weight': self.classifier.classifier.spectral_weight,
            'adaptive_threshold_enabled': self.classifier.classifier.use_adaptive_threshold,
            'adaptive_threshold_calculated': self.classifier.classifier.adaptive_threshold_calculated
        }
#this is used to load the audio from the directories
def _load_audio_from_dirs(male_dir: str, female_dir: str) -> Tuple[List[str], List[str]]:
    audio_paths, labels = [], []#initialize the audio paths and labels lists
    if os.path.exists(male_dir):#if the male directory exists, then:
        male_files = find_audio_files(male_dir)#find the audio files in the male directory
        audio_paths.extend(male_files)#extend the audio paths list with the male files
        labels.extend(['male'] * len(male_files))#extend the labels list with the male labels
    if os.path.exists(female_dir):#if the female directory exists, then:
        female_files = find_audio_files(female_dir)#find the audio files in the female directory
        audio_paths.extend(female_files)#extend the audio paths list with the female files
        labels.extend(['female'] * len(female_files))#extend the labels list with the female labels
    return audio_paths, labels if audio_paths else (None, None)#return the audio paths and labels if the audio paths are not None, otherwise return None
#this is basically used to load the audio from the filenames
def _load_audio_from_filenames(directory: str) -> Tuple[List[str], List[str]]:
    all_files = find_audio_files(directory)#find the audio files in the directory
    if not all_files:#if the audio files are not found, then:
        return None, None#return None if the audio files are not found
    audio_paths, labels = [], []#initialize the audio paths and labels lists
    for file_path in all_files:#iterate through the audio files
        filename = os.path.basename(file_path)#get the filename from the file path
        audio_paths.append(file_path)#append the file path to the audio paths list
        gender = infer_gender_from_filename(filename)#infer the gender from the filename
        labels.append(gender)#append the gender to the labels list
    return (audio_paths, labels) if audio_paths else (None, None)#return the audio paths and labels if the audio paths are not None, otherwise return None
#this is used to load the training data
def load_training_data():
    voice_samples_dir = get_voice_samples_dir()#get the voice samples directory
    audio_samples_dir = get_audio_samples_dir()#get the audio samples directory
    training_dir = os.path.join(audio_samples_dir, "training")#get the training directory
    audio_paths, labels = _load_audio_from_dirs(#load the audio from the directories
        os.path.join(voice_samples_dir, "Male_Voices"),
        os.path.join(voice_samples_dir, "Female_Voices")
    )
    if audio_paths:#if the audio paths is not empty basically, then:
        return audio_paths, labels#return the audio paths and labels
    audio_paths, labels = _load_audio_from_dirs(#load the audio from the directories
        os.path.join(training_dir, "male"),#get the male training directory
        os.path.join(training_dir, "female")#get the female training directory
    )
    if audio_paths:#if the audio paths is not empty basically, then:
        return audio_paths, labels#return the audio paths and labels
    if os.path.exists(voice_samples_dir):#if the voice samples directory exists, then:
        return _load_audio_from_filenames(voice_samples_dir)#load the audio from the filenames
    if os.path.exists(training_dir):#if the training directory exists, then:
        return _load_audio_from_filenames(training_dir)#load the audio from the filenames
    #return None if the audio paths and labels are not found
    return None, None
#this is used to load the test data
def load_test_data():
    voice_samples_dir = get_voice_samples_dir()#get the voice samples directory
    audio_samples_dir = get_audio_samples_dir()#get the audio samples directory
    test_dir = os.path.join(audio_samples_dir, "test")#get the test directory

    from config import MALE_VOICES_DIR, FEMALE_VOICES_DIR, PERSONAL_VOICES_DIR#use the config file to get the male voices directory, female voices directory, and personal voices directory
    audio_paths, labels = _load_audio_from_dirs(
        os.path.join(voice_samples_dir, MALE_VOICES_DIR),#get the male voices directory
        os.path.join(voice_samples_dir, FEMALE_VOICES_DIR)#get the female voices directory
    )
    if audio_paths:#if the audio paths is not empty basically, then:
        return audio_paths, labels#return the audio paths and labels

    personal_voices_dir = os.path.join(voice_samples_dir, PERSONAL_VOICES_DIR)#from config file as previously mentioned, this is the personal voices directory
    if os.path.exists(personal_voices_dir):#if the personal voices directory exists, then:
        audio_paths, labels = _load_audio_from_filenames(personal_voices_dir)#load the audio from the filenames
        if audio_paths:#if the audio paths is not empty basically, then:
            return audio_paths, labels#return the audio paths and labels

    audio_paths, labels = _load_audio_from_dirs(
        os.path.join(test_dir, "male"),#get the male test directory
        os.path.join(test_dir, "female")#get the female test directory
    )
    if audio_paths:#if the audio paths is not empty basically, then:
        return audio_paths, labels#return the audio paths and labels

    if os.path.exists(test_dir):#if the test directory exists, then:
        audio_paths, labels = _load_audio_from_filenames(test_dir)#load the audio from the filenames
        if audio_paths:#if the audio paths is not empty basically, then:
            return audio_paths, labels#return the audio paths and labels

    return None, None#return None if the audio paths and labels are not found

def main(): #this is the main function that is used to run the gender identification system (everything will be wired up here and ran from here)
    ensure_directory_exists(LOGGING_DIR)#ensure the logging directory exists
    print("Gender Identification Through Audio Analysis")#print the gender identification through audio analysis message
    print(f"\nLogging directory: {LOGGING_DIR}/")#print the logging directory
    system = GenderIdentificationSystem()#initialize the gender identification system
    classifier_info = system.get_classifier_info()#get the classifier information
    #print the classifier configuration
    print(f"\nClassifier Configuration:")
    print(f"  Method: {classifier_info['method']}")
    print(f"  Pitch Threshold: {classifier_info['pitch_threshold']:.2f} Hz", end="")
    if classifier_info.get('adaptive_threshold_calculated', False):#if the adaptive threshold is calculated, then:
        #print the adaptive threshold message
        print(" (Adaptive - calibrated from training data)")
    else:#if the adaptive threshold is not calculated, then:
        #print the default message
        print(" (Default)")
    #print the segment-based weights
    print(f"  Segment-based weights:")
    print(f"    Mean Pitch Weight: {classifier_info['mean_pitch_weight']:.2f}")
    print(f"    Pitch Std Dev Weight: {classifier_info['pitch_std_weight']:.2f}")
    print(f"  Overall feature weights:")
    print(f"    Average Frequency Weight: {classifier_info['avg_freq_weight']:.2f}")
    print(f"    Peak Frequency Weight: {classifier_info['peak_freq_weight']:.2f}")
    print(f"    Frequency Std Dev Weight: {classifier_info['freq_std_weight']:.2f}")
    print(f"  Spectral features (tone quality/voice texture):")
    print(f"    Spectral Weight: {classifier_info.get('spectral_weight', 0):.2f}")
    if classifier_info.get('adaptive_threshold_enabled', False):#if the adaptive threshold is enabled, then:
        #print the adaptive threshold message
        print(f"  Adaptive Threshold: Enabled")
    #print the loading training data message
    print("\nLOADING TRAINING DATA (Optional - for analysis)")
    #load the training data
    training_audio_paths, training_labels = load_training_data()
    #if the training audio paths and training labels are not empty basically, then:
    if training_audio_paths and training_labels:
        print(f"\nFound {len(training_audio_paths)} training samples:")
        print(f"  Male: {sum(1 for l in training_labels if l == 'male')}")
        print(f"  Female: {sum(1 for l in training_labels if l == 'female')}")

        try:#analyze the training data
            system.analyze_training_data(training_audio_paths, training_labels)#calling the analyze_training_data method
        except Exception as e:#if an exception is raised, then:
            #print the warning message
            print(f"\nWarning: Error during training data analysis: {e}")
            print("Continuing with feature analysis...")
        try:#analyze the training data
            import sys#circular import avoidance
            import os#circular import avoidance
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data_Logic'))#circular import avoidance
            from src.Data_Logic.feature_analyzer import FeatureAnalyzer
            analyzer = FeatureAnalyzer()#initialize the feature analyzer
            print("\nRunning feature analysis...")#print the running feature analysis message
            analyzer.analyze_batch(training_audio_paths, training_labels)#calling the analyze_batch method
            analyzer.save_results_csv(os.path.join(LOGGING_DIR, "feature_analysis_results.csv"))#calling the save_results_csv method
            print(f"Feature analysis saved to {LOGGING_DIR}/feature_analysis_results.csv")#print the feature analysis saved message
        except Exception as e:#if an exception is raised, then:
            #print the warning message
            print(f"\nWarning: Error during feature analysis: {e}")
            print("Continuing with test phase...")
    else:#if the training audio paths and training labels are empty basically, then:
        #print the no training data found message
        print("\nNo training data found (optional).")
        print("The rule-based classifier will work without training data.")
        print("\nTo analyze training data, organize files in:")
        print("  Voice_Samples/Male_Voices/")
        print("  Voice_Samples/Female_Voices/")
        print("\nOR")
        print("  audio_samples/training/male/")
        print("  audio_samples/training/female/")

    print("\nLOADING TEST DATA")
#load the test data
    test_audio_paths, test_labels = load_test_data()

    if not test_audio_paths:#if the test audio paths are not empty, then:
        #print the no test data found message
        print("\nNo test data found!")
        from config import MALE_VOICES_DIR, FEMALE_VOICES_DIR, PERSONAL_VOICES_DIR#from config file as previously mentioned, this is the male voices directory, female voices directory, and personal voices directory
        print("\nTo test the system, place test audio files in:")#print the test audio files directory
        print(f"  Voice_Samples/{MALE_VOICES_DIR}/")
        print(f"  Voice_Samples/{FEMALE_VOICES_DIR}/")
        print(f"  Voice_Samples/{PERSONAL_VOICES_DIR}/")
        print("\nOR")
        print("  audio_samples/test/")
        return#return if the test audio paths are not empty

    print(f"\nFound {len(test_audio_paths)} test samples")
    if test_labels:#if the test labels are not empty, then:
        #print the test labels message
        print(f"  With labels: {sum(1 for l in test_labels if l is not None)}")

    try:#test the samples
        results = system.test_samples(test_audio_paths, test_labels)
        #save the results to a csv file
        system.save_results_csv(results, os.path.join(LOGGING_DIR, "results.csv"))#calling the save_results_csv method
        #let the user know and print the process complete message
        print("\nPROCESS COMPLETE")#print the process complete message
        print(f"\nAll logs and results saved to '{LOGGING_DIR}/' directory:")#print the logs and results saved message
        print(f"  - results.csv (test results)")#print the test results message
        if training_audio_paths and training_labels:#if the training audio paths and training labels are not empty, then:
            print(f"  - feature_analysis_results.csv (feature analysis)")#print the feature analysis message
        print("\nClassifier uses custom rule-based method with:")#print the classifier uses custom rule-based method message
        print("  - 1-second segment analysis")#print the 1-second segment analysis message
        print("  - Weighted scoring (mean pitch + pitch std dev)")#print the weighted scoring message
        print("  - Majority voting across segments")#print the majority voting across segments message
    except Exception as e:#if an exception is raised, then:
        print(f"\nError during testing: {e}")#print the error during testing message
if __name__ == "__main__":#if the main function is run directly, then:
    main()
