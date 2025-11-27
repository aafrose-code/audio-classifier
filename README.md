# Gender Identification Through Audio Analysis - Project Documentation

## SETUP AND EXECUTION

### Virtual Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have audio files ready:
   - Minimum 10 audio recordings (at least 5 male, 5 female) for training
   - Each audio sample must be at least 10 seconds long
   - Supported formats: WAV, MP3, FLAC, etc. (librosa supports many formats)

### Execution

Run the main script:
```bash
python "src/Code Logic/main.py"
```

The script will:
- Look for training data in: `audio_samples/training/` (optional, for analysis)
- Look for test data in: `audio_samples/test/`
- Analyze training data if available
- Test on test samples using custom rule-based classifier
- Save results to `results/` directory

Organize your audio files:
```
audio_samples/
├── training/
│   ├── male/     (place male voice samples here)
│   └── female/   (place female voice samples here)
└── test/         (place test samples here)
```

### Running Tests

Run all test cases:
```bash
python "src/Test Cases/run_all_tests.py"
```

Run specific test class:
```bash
python -m unittest "src.Test Cases.test_comparison.TestVoiceSampleComparison" -v
```

## ROLE OF EACH FILE

### src/Code Logic/extract_voice_metadata.py
- `VoiceFeatureExtractor` class
- Extracts acoustic features from audio files
- Splits audio into 1-second segments
- Extracts mean pitch and pitch standard deviation per segment
- Handles audio loading and fundamental frequency extraction using librosa

### src/Code Logic/classification.py
- `CustomRuleBasedClassifier` class: Implements custom rule-based gender classification
  * Computes weighted scores for each segment (mean pitch weight: 0.7, pitch std dev weight: 0.3)
  * Classifies each segment as male or female based on pitch threshold (165 Hz)
  * Uses majority voting across segments for final gender prediction
- `GenderClassifier` class: Wrapper interface for the custom classifier

### src/Code Logic/main.py
- `GenderIdentificationSystem` class: Main orchestrator
  * Coordinates feature extraction and classification
  * Handles training data analysis (optional)
  * Processes test samples and generates results
  * Saves results to JSON and CSV formats
- Helper functions:
  * `find_audio_files()`: Finds audio files in directories
  * `load_training_data()`: Loads training data from directory structure
  * `load_test_data()`: Loads test data from directory structure
  * `main()`: Entry point that runs the complete workflow

### src/Test Cases/
- `test_feature_extraction.py`: Tests for `VoiceFeatureExtractor` class
- `test_classification.py`: Tests for `CustomRuleBasedClassifier` and `GenderClassifier` classes
- `test_system.py`: Tests for `GenderIdentificationSystem` class
- `test_helpers.py`: Tests for helper functions
- `test_comparison.py`: Voice sample comparison test with hardcoded expected results
- `run_all_tests.py`: Test runner to execute all tests

### requirements.txt
- Lists Python package dependencies:
  * numpy: Numerical computations
  * librosa: Audio analysis and feature extraction
  * soundfile: Audio file I/O

### README.md
- Project documentation (this file)

## FINDINGS

### Classification Method
The system uses a custom rule-based classifier that:
- Splits audio into 1-second segments for granular analysis
- Extracts mean pitch and pitch standard deviation from each segment
- Computes weighted scores (70% mean pitch, 30% pitch std dev)
- Classifies segments using a pitch threshold of 165 Hz
- Determines final gender through majority voting across all segments

### Key Parameters
- Pitch Threshold: 165 Hz (configurable)
- Mean Pitch Weight: 0.7 (70%)
- Pitch Std Dev Weight: 0.3 (30%)
- Segment Duration: 1 second

### Advantages
- Robust to temporary pitch variations within audio samples
- Handles audio with mixed characteristics throughout the recording
- Provides segment-level insights for analysis
- No training data required (rule-based approach)
- Majority voting makes the classifier resilient to outliers

### Limitations
- Requires minimum 10 seconds of audio per sample
- Performance depends on audio quality and clear speech
- Threshold-based approach may need tuning for specific populations
- Does not account for other acoustic features beyond pitch

### Test Results
- Test suite covers all code lines with comprehensive test cases
- Comparison test validates results against expected outcomes
- All core functionality tested including edge cases and error handling

