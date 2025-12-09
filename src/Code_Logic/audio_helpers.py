import os
from typing import List, Tuple
from config import SUPPORTED_AUDIO_EXTENSIONS
#finds the audio files in the directory using the os library
def find_audio_files(directory: str, extensions: Tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS) -> List[str]:
    audio_files = []
    if not os.path.exists(directory):
        return audio_files

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(root, file))

    return sorted(audio_files)
#verify if the audio file is actually an audio file with supported extensions
def is_audio_file(file_path: str, extensions: Tuple[str, ...] = SUPPORTED_AUDIO_EXTENSIONS) -> bool:
    return os.path.isfile(file_path) and file_path.lower().endswith(extensions)
#get the expected gender from the filename for testing purposes
def infer_gender_from_filename(filename: str) -> str:
    filename_lower = filename.lower()
    if 'male' in filename_lower or 'm_' in filename_lower:
        return 'male'
    elif 'female' in filename_lower or 'f_' in filename_lower or 'woman' in filename_lower:
        return 'female'
    return None