# trainCLI.py

# === MOCKABLE PLACEHOLDERS (patch or override during testing) ===
Train = object
Deploy = object
deploy_cli = lambda deploy: print("📦 [deploy_cli called]")
getNewModel = lambda: "MockModel"
getNewTokenizer = lambda: "MockTokenizer"
hyper_params = {"lr": 0.001, "batch_size": 32}
task_type = "classification"
browseDict = {"eval matrice": "Accuracy"}
init_model = lambda data, model: None


class BojaiTrainingCLI:
    def __init__(self, train):
        self.train = train
        self.trained = False

    def print_header(self, title):
        print("\n" + "=" * 60)
        print(f"{title}")
        print("=" * 60)

    def display_hyperparams(self):
        self.print_header("📊 Current Hyperparameters")
        for name, val in self.train.trainerManager.hyperparams.items():
            print(f"{name}: {val}")
        print(f"Device: {self.train.prep.device}")

    def update_hyperparams(self):
        self.print_header("🔧 Update Hyperparameters")
        new_params = {}
        for name, old_val in self.train.trainerManager.hyperparams.items():
            val = input(f"{name} (current: {old_val}) → ").strip()
            if val == "":
                new_params[name] = old_val
            else:
                try:
                    new_params[name] = float(val) if "." in val else int(val)
                except ValueError:
                    print(f"❌ Invalid value for {name}. Keeping old value.")
                    new_params[name] = old_val
        try:
            self.train.edit_hyperparams(new_params)
            print("✅ Hyperparameters updated.")
        except Exception as e:
            print(f"❌ Failed to update hyperparameters: {str(e)}")

    def train_model(self):
        self.print_header("🚀 Starting Training")
        model_class = self.train.model.__class__.__name__
        if model_class == "kNN":
            print("⚠️  kNN does not require training. You may proceed to evaluation.")
            return

        trainer = self.train.trainerManager.trainer

        try:

            class ProgressCallback:
                def emit(self, progress):
                    print(f"🟣 Progress: {progress}%", end="\r")

            class LossCallback:
                def emit(self, loss):
                    print(f"💥 Loss: {loss:.4f}", end="\r")

            class DummyQThread:
                def msleep(self, _):
                    pass

            trainer.train(DummyQThread(), ProgressCallback(), LossCallback())
            print("\n✅ Training complete.")
            self.trained = True
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")

    def evaluate_model(self):
        self.print_header("📈 Model Evaluation")
        try:
            score = self.train.trainerManager.trainer.evaluate()
            print(f"✅ Evaluation Result: {browseDict['eval matrice']} = {score}")
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")

    def replace_model(self):
        self.print_header("🔄 Replace Model")
        confirm = (
            input("Are you sure? This will reset your model. (y/n): ").strip().lower()
        )
        if confirm != "y":
            print("❌ Cancelled.")
            return
        try:
            self.train.trainerManager.trainer.model = getNewModel()
            init_model(self.train.prep.data, self.train.trainerManager.trainer.model)
            print("✅ Model successfully reset.")
        except Exception as e:
            print(f"❌ Could not replace model: {str(e)}")

    def init_deploy(self):
        deploy = Deploy(self.train, 100)
        deploy_cli(deploy)

    def run(self):
        self.print_header("🧠 Bojai Training CLI")

        while True:
            self.display_hyperparams()
            print("\nAvailable actions:")
            print("  [t] Start training")
            print("  [u] Update hyperparameters")
            print("  [e] Evaluate model")
            print("  [r] Replace model")
            print("  [d] Deploy model (if trained)")
            print("  [p] ⬅️  Go back to data preparation")
            print("  [q] Quit")

            choice = input("Enter your choice: ").strip().lower()

            if choice == "t":
                self.train_model()
            elif choice == "u":
                self.update_hyperparams()
            elif choice == "e":
                self.evaluate_model()
            elif choice == "r":
                self.replace_model()
            elif choice == "d":
                if not self.trained:
                    print("❌ Please train your model first before deploying.")
                else:
                    self.init_deploy()
            elif choice == "p":
                try:
                    from prepareCLI import BojaiPreparationCLI

                    print("🔁 Returning to data preparation CLI...")
                    BojaiPreparationCLI(self.train.prep).run()
                    print("✅ Returned to training stage.")
                except Exception as e:
                    print(f"❌ Failed to return: {e}")
            elif choice == "q":
                print("👋 Exiting training stage.")
                return
            else:
                print("❌ Invalid choice.")
