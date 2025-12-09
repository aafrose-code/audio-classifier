import csv
import os
from typing import List, Dict
def save_to_csv(data: List[Dict], output_path: str, fieldnames: List[str] = None):#save the data to a csv file
    if not data:#if the data is not empty, then:
        print(f"Warning: No data to save to {output_path}")#print the warning message to the user
        return#return if the data is not empty
    output_dir = os.path.dirname(output_path)#get the output directory
    if output_dir and not os.path.exists(output_dir):#if the output directory is not set and the output directory does not exist, then:
        os.makedirs(output_dir)#create the output directory
    if fieldnames is None:#if the fieldnames are not set, then:
        fieldnames = list(data[0].keys())#set the fieldnames to the keys of the data
    try:#save the data to a csv file
        with open(output_path, 'w', newline='', encoding='utf-8') as f:#encode the data to a utf-8 format
            writer = csv.DictWriter(f, fieldnames=fieldnames)#create a new writer
            writer.writeheader()#write the header to the csv file
            writer.writerows(data)#write the data to the csv file
        print(f"Results saved to {output_path}")#print the results saved message to the user
    except Exception as e:#if an exception is raised, then:
        print(f"Error saving to {output_path}: {e}")#print the error message to the user
        raise#raise the exception

