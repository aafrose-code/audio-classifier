import os
from typing import List, Dict
from datetime import datetime
from utils import save_to_csv
class DataCollectionLogger:#data collection logger class used to log the data collection process
    def __init__(self, output_dir: str = None):#initialize the data collection logger
        self.output_dir = output_dir#set the output directory
        self.collected_data = []#initialize the collected data list
    def log_sample(self, audio_path: str, label: str = None, metadata: Dict = None):#log a sample of the collected data
        entry = {#create a new entry for the collected data
            'timestamp': datetime.now().isoformat(),
            'audio_path': audio_path,
            'label': label,
            'filename': os.path.basename(audio_path)
        }
        if metadata:#if the metadata is not empty, then:
            entry.update(metadata)#update the entry with the metadata
        self.collected_data.append(entry)#add the entry to the collected data list
    def log_batch(self, audio_paths: List[str], labels: List[str] = None):#log a batch of the collected data
        for i, audio_path in enumerate(audio_paths):#loop through the audio paths
            label = labels[i] if labels and i < len(labels) else None#get the label for the audio path
            self.log_sample(audio_path, label)#log the sample
    def save_log(self, output_path: str = None):#save the log to a csv file
        if output_path is None and self.output_dir:#if the output path is not set and the output directory is set, then:
            output_path = os.path.join(self.output_dir, 'data_collection_log.csv')#set the output path to the data collection log csv file
        if output_path and self.collected_data:#if the output path is set and the collected data is not empty, then:
            save_to_csv(self.collected_data, output_path)#save the collected data to a csv file

