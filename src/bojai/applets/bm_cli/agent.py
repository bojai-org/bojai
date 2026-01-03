from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ollama
import re
import shutil
import os

class PromptError(Exception):
    pass

class WriteError(Exception):
    pass

class ImplementationError(Exception):
    pass

class DescriptionError(Exception):
    pass


class DataProcessorAgent:
    def __init__(self, address, data_dir, model_name="mistral") -> None:
        self.tokenizer = None
        self.model = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.address = address
        self.data_dir = data_dir

    def process(self, description, image):
        # Create a temporary copy
        temp_file = self.address.replace(".py", "_temp.py")
        shutil.copy2(self.address, temp_file)
        original_address = self.address
        self.address = temp_file  # Point agent to work on temp file

        max_tries = 3
        tries = 0
        feedback = ""

        while tries < max_tries:
            try:
                answer = self.ask_local_model(description, image, feedback)
                new_func_def = self.write_function(answer)
                self.test_function()

                # Success: replace original file with modified temp
                shutil.move(temp_file, original_address)
                self.address = original_address
                print("Successfully updated the processor.")
                return
            except (PromptError, WriteError) as e:
                tries += 1
                if tries >= max_tries:
                    print("Max tries reached. Cleaning up.")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    self.address = original_address
                    raise DescriptionError(
                        "The LLM was unable to implement your function. Try changing the description or implement it yourself."
                    ) from e

            except Exception as e:
                tries += 1
                feedback = f"Your previous implementation:\n{new_func_def}\nis wrong. Error:\n{str(e)}"
                print(f"Error: {e}")
                if tries < max_tries:
                    print("Trying again...")
                else: 
                    print("Max tries reached. Cleaning up.")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    self.address = original_address
                    raise DescriptionError(
                        "The LLM was unable to implement your function. Try changing the description or implement it yourself."
                    ) from e

        # Fallback cleanup if something else breaks
        if os.path.exists(temp_file):
            os.remove(temp_file)
        self.address = original_address


    def write_function(self, reply):
        match = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
        if match:
            function_text = match.group(1).strip()
        else:
            raise PromptError("Prompt did not output a function.")

        try:
            with open(self.address, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Find start of function
            start_idx = None
            for i, line in enumerate(lines):
                if re.match(r"\s*def get_inputs_outputs\s*\(self,\s*data_dir\):", line):
                    start_idx = i
                    break

            if start_idx is None:
                raise WriteError("Function definition not found.")

            # Find line that includes 'return' and is indented
            end_idx = None
            for i in range(start_idx + 1, len(lines)):
                if 'return' in lines[i] and lines[i].startswith((' ' * 4, '\t')):
                    end_idx = i
                    break

            if end_idx is None:
                raise WriteError("No return statement found in the function.")

            # Replace that block
            new_lines = function_text.strip().splitlines()
            indent = re.match(r'^(\s*)', lines[start_idx]).group(1)
            indented_new_func = [indent + line + '\n' for line in new_lines]

            updated_lines = (
                lines[:start_idx] +
                indented_new_func +
                lines[end_idx + 1:]
            )

            with open(self.address, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            return ''.join(indented_new_func)

        except Exception as e:
            raise WriteError(f"Error writing the function: {e}")



    def test_function(self):
        from custom_data_processor_temp import YourDataProcessor
        processor = YourDataProcessor(self.data_dir, [0.5, 0.5], self.model, None, self.tokenizer)
        processor.get_item_untokenized(0)
            

    def ask_local_model(self, description, image, feedback):
        prompt = f'''
            You are a precise developer who can write data processors well. You are asked to code the initial part of data processing. Carefully follow instructions below.

            Here is how my data is structured:
            {description}

            Write a function get_inputs_outputs(self, data_dir) that returns the needed inputs and outputs from my data as described above. Make it return two lists inputs, outputs. 

            {"Images are involved; the outputs list is just the address of each image." if image else ""} 

            If you use any import statements, import them inside the function definition. Use the simplest way possible to implement, only import libraries that are absolutely needed. 

            Return the function in a .md python code block python``` code ```. The function should look like this, do not include the class definition, just the function: 

            python ```
            def get_inputs_outputs(self, data_dir): 
                import x #only if needed 
                from x import y # only if needed, not necessairy
                # function body
                # your code here use data_dir parameter to read based on description above
                inputs = # your code
                outputs = # your code
                # your code
                return inputs, outputs
            ```

            {feedback}
        '''

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']