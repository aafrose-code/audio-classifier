import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class VoiceFeatureExtractor:
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=None)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {str(e)}")
    
    def extract_fundamental_frequency(self, audio: np.ndarray) -> np.ndarray:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        
        f0_clean = f0[~np.isnan(f0)]
        
        return f0_clean
    
    def extract_segment_features(self, audio: np.ndarray, segment_duration: float = 1.0) -> List[Dict[str, float]]:
        sr = self.sample_rate
        segment_samples = int(segment_duration * sr)
        num_segments = len(audio) // segment_samples
        
        if num_segments == 0:
            raise ValueError(f"Audio too short for {segment_duration}-second segments")
        
        segment_features = []
        
        for i in range(num_segments):
            start_idx = i * segment_samples
            end_idx = start_idx + segment_samples
            segment_audio = audio[start_idx:end_idx]
            
            f0 = self.extract_fundamental_frequency(segment_audio)
            
            if len(f0) > 0:
                mean_pitch = float(np.mean(f0))
                pitch_std = float(np.std(f0))
                
                segment_features.append({
                    'segment_index': i,
                    'mean_pitch': mean_pitch,
                    'pitch_std': pitch_std
                })
        
        return segment_features
    
    def extract_features(self, audio_path: str, use_segments: bool = True) -> Dict:
        audio, sr = self.load_audio(audio_path)
        
        min_duration = 10.0
        min_samples = int(min_duration * sr)
        
        if len(audio) < min_samples:
            raise ValueError(
                f"Audio file must be at least {min_duration} seconds long. "
                f"Current duration: {len(audio) / sr:.2f} seconds"
            )
        
        if use_segments:
            segments = self.extract_segment_features(audio)
            
            if len(segments) == 0:
                raise ValueError("No valid segments found. Audio may not contain speech.")
            
            features = {
                'segments': segments,
                'num_segments': len(segments)
            }
        else:
            f0 = self.extract_fundamental_frequency(audio)
            
            if len(f0) == 0:
                raise ValueError("No valid fundamental frequency detected. Audio may not contain speech.")
            
            features = {
                'average_frequency': float(np.mean(f0)),
                'peak_frequency': float(np.max(f0)),
                'frequency_std': float(np.std(f0))
            }
        
        return features
    
    def extract_features_batch(self, audio_paths: list, use_segments: bool = True) -> list:
        results = []
        for audio_path in audio_paths:
            try:
                features = self.extract_features(audio_path, use_segments=use_segments)
                features['audio_path'] = audio_path
                results.append(features)
            except Exception as e:
                print(f"Warning: Failed to extract features from {audio_path}: {str(e)}")
                continue
        
        return results
