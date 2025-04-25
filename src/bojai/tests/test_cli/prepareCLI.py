# prepareCLI.py

class BojaiPreparationCLI:
    def __init__(
        self,
        prep,
        train_cls=None,
        train_cli_fn=None,
        browse_dict=None,
        hyperparams=None
    ):
        self.prep = prep
        self.Train = train_cls or (lambda prep, hparams: "MockTrainObject")
        self.train_cli = train_cli_fn or (lambda train: print("📦 [train_cli called]"))
        self.browseDict = browse_dict or {"type": 2}
        self.hyper_params = hyperparams or {"lr": 0.001}

    def print_header(self, title):
        print("\n" + "=" * 60)
        print(f"{title}")
        print("=" * 60)

    def display_pipeline_info(self):
        self.print_header("📄 Pipeline Information")
        print(f"🔤 Model Name        : {self.prep.model_name}")
        print(f"🧠 Model Type        : {type(self.prep.model).__name__}")
        print(f"🔡 Tokenizer Type    : {type(self.prep.tokenizer).__name__}")
        print(f"📈 Data Points       : {self.prep.num_data_points}")
        print(f"✅ Prep Ready?       : {'Yes' if self.prep.prep_ready else 'No'}")

    def view_tokenized_data(self):
        while True:
            index = input("Enter index to view tokenized data (leave blank for a random data point, q to quit): ").strip()
            if index == "q":
                break
            try:
                if index == "":
                    print(self.prep.check_tokenized())
                elif index.isdigit():
                    print(self.prep.check_tokenized(int(index) - 1))
                else:
                    print("❌ Invalid input.")
            except IndexError:
                print("❌ Index too large.")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def view_raw_data(self):
        while True:
            index = input("Enter index to view raw data (leave blank for all, q to quit): ").strip()
            if index == "q":
                break
            try:
                data_type = self.browseDict.get("type", 2)
                if index == "":
                    output = self.prep.view_raw_data()
                elif index.isdigit():
                    output = self.prep.view_raw_data(int(index) - 1)
                else:
                    print("❌ Invalid input.")
                    continue

                if data_type == 0:
                    image, label = output
                    print(f"[🖼️ Image Data]: {label}")
                    image.show()
                else:
                    print(f"[📃 Raw]: {output}")

            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def update_data_path(self):
        new_path = input("Enter new dataset address: ").strip()
        if not new_path:
            print("❌ Path cannot be empty.")
            return
        try:
            self.prep.update_data(new_path)
            print(f"✅ Data updated successfully. Total data points: {self.prep.num_data_points}")
        except Exception as e:
            print(f"❌ Failed to update data: {str(e)}")

    def do_training(self):
        try:
            trainer = self.Train(self.prep, self.hyper_params)
            print("🧠 Launching training CLI...")
            self.train_cli(trainer)
        except Exception as e:
            print(f"❌ Failed to launch training: {str(e)}")

    def run(self):
        self.print_header("🔧 Bojai Data Preparation CLI")
        while True:
            self.display_pipeline_info()
            print("\nAvailable actions:")
            print("  [v] View tokenized data")
            print("  [r] View raw data")
            print("  [u] Update dataset path")
            print("  [t] Continue to training")
            print("  [q] Quit")

            choice = input("Enter your choice: ").strip().lower()

            if choice == "v":
                self.view_tokenized_data()
            elif choice == "r":
                self.view_raw_data()
            elif choice == "u":
                self.update_data_path()
            elif choice == "t":
                if not self.prep.prep_ready:
                    print("❌ Prep is not ready. Cannot continue to training.")
                    continue
                print("✅ Prep complete. You can now continue to training stage.")
                return self.do_training()
            elif choice == "q":
                print("👋 Exiting data preparation stage.")
                return
            else:
                print("❌ Invalid option. Try again.")