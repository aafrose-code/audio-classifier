# Gender Identification Through Audio Analysis

A simple program that listens to voice recordings and guesses whether the speaker is male or female.

## What You Need

- Python installed on your computer
- At least 10 audio files (5 male voices, 5 female voices)
- Each audio file should be at least 10 seconds long with someone actually talking
- Supported formats: WAV, MP3, FLAC, M4A, OGG

## Quick Start

### 1. Set Up Virtual Environment (Recommended)

A virtual environment keeps your project's packages separate from other projects. It's like having a separate toolbox for this project.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You'll know it worked when you see `(venv)` at the start of your command line.

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Put Your Audio Files in the Right Place

Create this folder structure:

```
Voice_Samples/
├── Male_Voices/     (put male voice files here)
└── Female_Voices/   (put female voice files here)
```

### 4. Run the Program

```bash
python "src/Code_Logic/main.py"
```

That's it! The program will:
- Analyze your audio files
- Guess if each voice is male or female
- Save the results to `src/logging/results.csv`

## How It Works (Simple Version)

The program looks at how high or low someone's voice is (called "pitch"). 

- **Male voices** are usually lower (like a bass guitar)
- **Female voices** are usually higher (like a violin)

The program:
1. Splits each audio file into 1-second chunks
2. Measures the pitch of each chunk
3. Votes on whether each chunk sounds male or female
4. The majority vote wins

## What Each File Does

Each file has a specific job:

### The Main Files (Code_Logic folder)

**`main.py`** - The boss file that runs everything
- This is the file you run to start the program
- It tells all the other files what to do and when
- It loads your audio files, runs the analysis, and saves the results

**`extract_voice_metadata.py`** - The listener
- Takes your audio files and "listens" to them
- Figures out how high or low the voice is (the pitch)
- Splits long recordings into 1-second chunks to analyze each part separately

**`classification.py`** - The decision maker
- Takes the pitch information from the listener
- Looks at each 1-second chunk and votes: "This sounds male" or "This sounds female"
- Counts all the votes and makes the final decision (majority wins!)

**`config.py`** - The settings file
- Stores all the important numbers (like the pitch threshold of 170 Hz)
- If you want to change how the program works, you edit this file
- Think of it like the control panel

**`path_helpers.py`** and **`audio_helpers.py`** - The helpers
- These find your audio files and make sure everything is in the right place
- They're like assistants that help the main files do their job

### The Data Files (Data_Logic folder)

**`feature_analyzer.py`** - The statistician
- Looks at all the voice features and calculates averages
- Shows you the differences between male and female voices
- Creates the detailed statistics you see in the results

**`data_collection_logger.py`** - The record keeper
- Keeps track of which audio files you used
- Remembers where each file came from
- Helps you organize your data

**`utils.py`** - The file saver
- Takes all the results and saves them to CSV files
- Makes sure your results are saved in a format you can open in Excel

### How They Work Together

1. **main.py** starts everything and loads your audio files
2. **extract_voice_metadata.py** listens to each file and gets the pitch
3. **classification.py** looks at the pitch and decides male or female
4. **feature_analyzer.py** calculates statistics about all the voices
5. **utils.py** saves everything to CSV files

## Results

All results are saved in the `src/logging/` folder:
- `results.csv` - Shows predictions for each audio file (was it right or wrong?)
- `feature_analysis_results.csv` - Shows detailed statistics about the voices (average pitch, etc.)

## Project Structure

```
src/
├── Code_Logic/          (Main program files)
│   └── main.py         (Run this file!)
├── Test_Cases/         (Tests to make sure everything works)
└── logging/            (Results saved here)
```
