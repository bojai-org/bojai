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
        return {'@TODO your-model-name' : '@TODO give some name to data name'}
    
        
    def get_dir_type(self, task, data_dir):
        '''
        @TODO
        implement a function to verify given data_dir is of expected type. This is not required. 



        '''
        self.data_dir = data_dir
        

