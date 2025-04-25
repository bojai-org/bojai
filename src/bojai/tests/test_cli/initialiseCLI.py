import argparse
import os


# TEMP PLACEHOLDERS (these get mocked in tests)
def getNewModel():
    pass


def getNewTokenizer():
    pass


def prepare_cli(prep):
    pass


browseDict = {"options": 0, "options-where": 0}
options = {"bert": lambda: "TokenizerInstance"}
task_type = "classification"


def prompt_input(prompt_text, cast_type=str, allow_blank=False):
    while True:
        value = input(prompt_text).strip()
        if not value and not allow_blank:
            print("❌ Input cannot be blank.")
            continue
        try:
            return cast_type(value)
        except ValueError:
            print(f"❌ Invalid input. Please enter a {cast_type.__name__}.")


def data_preparation_stage_cli(prep):
    print("\n────────────────────────────────────")
    print("📁 Entering Data Preparation Stage")
    print("────────────────────────────────────")
    prepare_cli(prep)


def initialize_pipeline_cli():
    print("📦 Welcome to Bojai CLI Initializer")
    print("────────────────────────────────────\n")

    model_name = prompt_input("Enter Model Name: ")

    data_address = prompt_input("Enter Data Address (file or directory): ")
    if not os.path.exists(data_address):
        print("❌ Path does not exist. Exiting.")
        return

    print("\n⚖️  Set the train/eval split (must add to 1)")
    train_split = prompt_input("Training split (e.g., 0.8): ", float)
    eval_split = prompt_input("Evaluation split (e.g., 0.2): ", float)

    if round(train_split + eval_split, 2) != 1.0:
        print("❌ Train and eval splits must add to 1.0. Exiting.")
        return

    architecture = None
    if browseDict["options"] == 1:
        print("\n🧠 Choose a tokenizer architecture:")
        arch_names = list(options.keys())
        for idx, name in enumerate(arch_names):
            print(f"{idx + 1}. {name}")
        selected = prompt_input("Enter number of architecture: ", int)
        if selected < 1 or selected > len(arch_names):
            print("❌ Invalid architecture number. Exiting.")
            return
        architecture = arch_names[selected - 1]

    try:
        if browseDict["options-where"] == 0:
            model = getNewModel()
            tokenizer = options[architecture]() if architecture else None
        elif browseDict["options-where"] == 1:
            model = options[architecture]() if architecture else None
            tokenizer = getNewTokenizer()
        else:
            model = getNewModel()
            tokenizer = getNewTokenizer()

        # DEVICE MOCKED
        device = "cpu"

        # MOCKED PREP
        prep = {"model_name": model_name, "data": data_address}

    except Exception as e:
        print(f"❌ Error initializing pipeline: {str(e)}")
        return

    print("\n✅ Pipeline successfully initialized!")
    print("────────────────────────────────────")
    print(f"Model Name       : {model_name}")
    print(f"Data Path        : {data_address}")
    print(f"Train/Eval Split : {train_split}/{eval_split}")
    print(f"Tokenizer        : {architecture if architecture else 'Default'}")
    print("────────────────────────────────────\n")

    choice = (
        input("Type 'p' to proceed with data preparation, or any other key to exit: ")
        .strip()
        .lower()
    )
    if choice == "p":
        data_preparation_stage_cli(prep)
    else:
        print("Exiting Bojai CLI. Goodbye!")


if __name__ == "__main__":
    initialize_pipeline_cli()
