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
        
        if task == "get":
            processor = DataFormaterGET(data_dir)
            processor.process(actual, data_dir, data_sep)
            self.data_dir = processor.data_dir
            return 

        
    #gets all the mappings of task type to format.
    def get_all(self):
        return {'get' : 'f2'}
    

    def count_sentences_in_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Split by periods (assuming sentences are separated by periods)
            sentences = text.split('\n')
            # Remove empty entries and count non-empty sentences
            return len([sentence for sentence in sentences if sentence.strip()])
        
    def get_dir_type(self, task, data_dir):
        print(data_dir[-4:])
        if os.path.isdir(data_dir):
        # List all files in the directory
            files = os.listdir(data_dir)
        
        # Filter the list to get only .txt files
            txt_files = [file for file in files if file.endswith('.txt')]
        
        # Check if there are exactly 2 .txt files
            if len(txt_files) == 2:
                file1 = os.path.join(data_dir, txt_files[0])
                file2 = os.path.join(data_dir, txt_files[1])
            
            # Count sentences in both files
                sentences_file1 = self.count_sentences_in_file(file1)
                sentences_file2 = self.count_sentences_in_file(file2)
            
            # Check if the number of sentences is equal
            if sentences_file1 == sentences_file2:
                return 'f2'
            else:
                raise Exception()
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

        return None       
        
        
        
            
            

#formats data based on the specific model. 
class DataFormater(ABC):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    @abstractmethod
    def process(self, actual, data_dir, data_sep = None):
        pass

#data processor for GET, reutrns a directory with two txt files one for input one for output. 
class DataFormaterGET(DataFormater):
    def __init__(self, data_dir):
        super().__init__(data_dir)
    
    def process(self, actual, data_dir, data_sep = None):
        if actual == "json":
            self.convertJson(data_dir)
        elif actual == "csv":
            self.convertCSV(data_dir)
        elif actual == "f1":
            self.convertF1(data_dir, data_sep)
        elif actual == "yaml":
            self.convertYAML(data_dir)
        elif actual == "xml":
            self.convertXML(data_dir)

        self.data_dir = "actual_data" #os.path.abspath(os.path.join(os.curdir, 'actual_data'))
    
    def convertJson(self, data_dir):
        # Your logic to convert JSON
        file_path = data_dir
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        os.makedirs('actual_data', exist_ok=True)
        # Create output directories for input and output files
        input_file = os.path.join('actual_data', 'input.txt')
        output_file = os.path.join('actual_data', 'output.txt')

        with open(input_file, 'w') as input_txt, open(output_file, 'w') as output_txt:
            for entry in data:
                input_txt.write(entry.get('input', '') + '\n')
                output_txt.write(entry.get('output', '') + '\n')

    def convertCSV(self, data_dir):
        # Your logic to convert CSV
        file_path = data_dir
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            os.makedirs('actual_data', exist_ok=True)
            input_file = os.path.join('actual_data', 'input.txt')
            output_file = os.path.join('actual_data', 'output.txt')

            with open(input_file, 'w') as input_txt, open(output_file, 'w') as output_txt:
                for row in reader:
                    input_txt.write(row['input'] + '\n')
                    output_txt.write(row['output'] + '\n')

    def convertF1(self, data_dir, data_sep):
        # Your logic to convert F1
        file_path = data_dir
        os.makedirs('actual_data', exist_ok=True)
        input_file = os.path.join('actual_data', 'input.txt')
        output_file = os.path.join('actual_data', 'output.txt')

        with open(file_path, 'r') as file:
            lines = file.readlines()

        with open(input_file, 'w') as input_txt, open(output_file, 'w') as output_txt:
            for line in lines:
                parts = line.split(data_sep) 
                if len(parts) == 2:
                    input_txt.write(parts[0].strip() + '\n')
                    output_txt.write(parts[1].strip() + '\n')

    def convertYAML(self, data_dir):
        # Your logic to convert YAML
        file_path = data_dir
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        os.makedirs('actual_data', exist_ok=True)
        input_file = os.path.join('actual_data', 'input.txt')
        output_file = os.path.join('actual_data', 'output.txt')

        with open(input_file, 'w') as input_txt, open(output_file, 'w') as output_txt:
            if isinstance(data, list):
                for entry in data:
                    input_txt.write(entry.get('input', '') + '\n')
                    output_txt.write(entry.get('output', '') + '\n')
            elif isinstance(data, dict):
                input_txt.write(data.get('input', '') + '\n')
                output_txt.write(data.get('output', '') + '\n')

    def convertXML(self, data_dir):
        file_path = data_dir
        tree = ET.parse(file_path)
        root = tree.getroot()

        os.makedirs('actual_data', exist_ok=True)
        input_file = os.path.join('actual_data', 'input.txt')
        output_file = os.path.join('actual_data', 'output.txt')

        with open(input_file, 'w') as input_txt, open(output_file, 'w') as output_txt:
            for entry in root.findall('entry'):
                input_data = entry.find('input').text
                output_data = entry.find('output').text
                input_txt.write(input_data + '\n')
                output_txt.write(output_data + '\n')

