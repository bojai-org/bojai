# deployCLI.py

# === MOCKABLE PLACEHOLDERS ===
torch = type("torch", (), {"save": lambda obj, path: print(f"[torch.save] {path}")})
browseDict = {"eval matrice": "Accuracy", "use_model_text": "Enter input for model:"}
getNewModel = lambda: "MockModel"
getNewTokenizer = lambda: "MockTokenizer"


class BojaiDeployCLI:
    def __init__(self, deploy, train_cli_fn=None, prepare_cli_fn=None):
        self.deploy = deploy
        self.train_cli = train_cli_fn or (lambda train: print("📦 [train_cli called]"))
        self.prepare_cli = prepare_cli_fn or (
            lambda prep: print("📦 [prepare_cli called]")
        )

    def print_header(self, title):
        print("\n" + "=" * 60)
        print(f"{title}")
        print("=" * 60)

    def update_eval_data_cli(self):
        self.print_header("📂 Add New Evaluation Data")
        path = input("Enter path to new evaluation data (file or directory): ").strip()
        if not path or not hasattr(self.deploy, "update_eval_data"):
            print("❌ Invalid path.")
            return
        try:
            self.deploy.update_eval_data(path)
            print("✅ New evaluation data added.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def evaluate_original_data(self):
        self.print_header("📈 Evaluate Model on Original Data")
        try:
            score = self.deploy.get_eval_score(0)
            print(f"✅ {browseDict['eval matrice']}: {score}")
        except Exception as e:
            print(f"❌ Failed to evaluate: {str(e)}")

    def evaluate_new_data(self):
        if getattr(self.deploy, "new_data", None) is None:
            print("⚠️  No new evaluation data found.")
            return
        self.print_header("📈 Evaluate Model on New Data")
        try:
            score = self.deploy.get_eval_score(1)
            print(f"✅ {browseDict['eval matrice']}: {score}")
        except Exception as e:
            print(f"❌ Failed to evaluate: {str(e)}")

    def use_model_cli(self):
        self.print_header("🧠 Use Model for Inference")
        text_input = input(f"{browseDict['use_model_text']}\n> ").strip()
        try:
            self.deploy.max_length = 50
            output = self.deploy.use_model(text_input)
            print(f"✅ Output: {output}")
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")

    def save_model_cli(self):
        self.print_header("💾 Save Model")
        path = input("Enter directory to save model: ").strip()
        if not path:
            print("❌ Path must be a valid directory.")
            return
        try:
            model_name = self.deploy.trainer.model.__class__.__name__
            save_path = f"{path}/{model_name}.bin"
            torch.save(self.deploy.trainer.model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")

    def run(self):
        self.print_header("🚀 Bojai Model Deployment CLI")
        while True:
            print("\nAvailable actions:")
            print("  [a] Add new evaluation data")
            print("  [e] Evaluate on original data")
            if getattr(self.deploy, "new_data", None):
                print("  [n] Evaluate on new data")
            print("  [u] Use model for inference")
            print("  [s] Save model")
            print("  [t] ⬅️  Go back to training")
            print("  [p] ⬅️  Go back to data preparation")
            print("  [q] Quit")

            choice = input("Enter your choice: ").strip().lower()

            if choice == "a":
                self.update_eval_data_cli()
            elif choice == "e":
                self.evaluate_original_data()
            elif choice == "n" and self.deploy.new_data:
                self.evaluate_new_data()
            elif choice == "u":
                self.use_model_cli()
            elif choice == "s":
                self.save_model_cli()
            elif choice == "t":
                print("🔁 Returning to training CLI...")
                self.train_cli(self.deploy.trainer)
                print("✅ Returned to deployment stage.")
            elif choice == "p":
                print("🔁 Returning to data preparation CLI...")
                self.prepare_cli(self.deploy.trainer.prep)
                print("✅ Returned to deployment stage.")
            elif choice == "q":
                print("👋 Exiting deployment CLI.")
                return
            else:
                print("❌ Invalid choice.")
