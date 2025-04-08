from abc import ABC, abstractmethod
import os
import json
import csv
import yaml
import xml.etree.ElementTree as ET



#manages data formatting 
class dataFormatManager():
    def __init__(self):
        self.all_mappings = self.get_all()
        self.data_dir = ''

    def __call__(self, task : str, data_dir : str, data_sep :str):
        self.process(task, data_dir, data_sep)
    
    #checks whether the task code matches the dir type, if not calls some type of dataFormater to format data
    def process(self, task, data_dir,data_sep): 
        expected = self.all_mappings[task]
        actual = self.get_dir_type(task, data_dir)

        if actual == expected: 
            self.data_dir =  data_dir
            return
        
        if actual == None:
            raise ValueError("not suitable format, please refer to documentation to find supported formats")

        
    #gets all the mappings of task type to format.
    def get_all(self):
        return {'cln' : 'json'}
    

    def count_sentences_in_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Split by periods (assuming sentences are separated by periods)
            sentences = text.split('\n')
            # Remove empty entries and count non-empty sentences
            return len([sentence for sentence in sentences if sentence.strip()])
        
    def get_dir_type(self, task, data_dir):
        if os.path.isdir(data_dir):
            input_dir = os.path.join(data_dir, "input")
            txt_file = os.path.join(data_dir, "output.txt")

            if os.path.isdir(input_dir) and os.path.isfile(txt_file):
                images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                with open(txt_file, 'r', encoding='utf-8') as file:
                    num_lines = sum(1 for _ in file)

                if num_lines == len(images):
                    return 'fit'
                
        elif data_dir[-3:] == "txt":
            return "f1"
        elif data_dir[-4:] == "json":
            return "json"  
        elif data_dir[-3:] == "csv":
            return "csv"
        elif data_dir[-4:] == "yaml":
            return "yaml"
        elif data_dir[-3:] == "xml":
            return "xml"       

        return "none"        
            
#formats data based on the specific model. 
class DataFormater(ABC):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    @abstractmethod
    def process(self, actual, data_dir, data_sep = None):
        pass


