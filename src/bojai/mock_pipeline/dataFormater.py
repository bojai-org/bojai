# ───────────────────────────────────────────────────────────────────────────────
# dataFormatManager: Optional adapter to reformat data directories into a usable format
# ───────────────────────────────────────────────────────────────────────────────
'''
📦 What is this?
────────────────
`dataFormatManager` is an *optional plug-in* used to reformat user-provided data into a format your pipeline understands. 
It is optional and no need to touch it if you are sure the users of your pipeline will have the expected data format. 

Imagine a user gives you:
- XML files instead of CSV
- Images in nested folders instead of flat
- JSON lines when you expect raw text

This class gives you a way to:
- Detect the format of the data folder
- Automatically convert it if needed
- Ensure your processor always sees the format it expects

You define:
- What formats are expected for each `task`
- How to detect the actual format of the data
- Optionally, how to convert from one to another

🎯 Goal: Let users bring "messy" data — you silently clean it up for them.
'''

class dataFormatManager():
    def __init__(self):
        # Define expected formats per model/task
        self.all_mappings = self.get_all()

        # Store the final usable path after any reformatting
        self.data_dir = ''

    def __call__(self, task: str, data_dir: str, data_sep: str):
        '''
        This allows the manager to be called like a function:
            formatter(task_type, "path/to/data", ",")
        '''
        self.process(task, data_dir, data_sep)

    def process(self, task, data_dir, data_sep):
        '''
        1. Looks up what data format is expected for the given task.
        2. Uses get_dir_type() to determine what format the provided data actually is.
        3. If they match, great — continue.
        4. If not, raise an error or auto-convert (if you've implemented that logic).

        🔧 You can extend this to:
        - Call formatters (like XML→CSV converters)
        - Clean filenames
        - Flatten directory structures
        '''
        expected = self.all_mappings[task]
        actual = self.get_dir_type(task, data_dir)

        if actual == expected:
            self.data_dir = data_dir
            return

        if actual is None:
            raise ValueError(
                "❌ Unsupported data format. Please refer to the documentation for supported formats."
            )

        # Optionally: Add reformatting logic here if actual != expected

    def get_all(self):
        '''
        🗺️ Define expected format labels for each model/pipeline/task.
        These are just strings — they can mean anything you want (e.g., 'csv', 'xml', 'foldered-images')

        Example:
            return {
                "summarizer": "csv",
                "digit_classifier": "flat_image_dir",
                "my_custom_pipeline": "jsonl"
            }
        '''
        return {'@TODO your-model-name': '@TODO give some name to data format'}

    def get_dir_type(self, task, data_dir):
        '''
        🕵️‍♂️ Inspect the provided data_dir and guess what kind of data it contains.

        This is 100% customizable — implement whatever logic fits your needs.
        For example:
        - Look for `.xml` files
        - Check if directory contains folders (for classification tasks)
        - Validate existence of `labels.csv` file

        You must return a string label that matches one of those defined in `get_all()`.
        If format is unknown, return None.

        NOTE: You do NOT have to implement this.
        If you don’t use the formatter, Bojai will just use raw data as-is.
        '''
        self.data_dir = data_dir 
