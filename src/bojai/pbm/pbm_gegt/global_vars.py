from model import FineTunedTransformerGEGT
import requests
from transformers import ViTImageProcessor, BertTokenizer

# file : 1 , dir : 0
browseDict = {
    "train": False,
    "prep": False,
    "deploy_new_data": False,
    "use_model_upload": True,  # if 1 means the use_model needs a browse, if not it doesn't.
    "use_model_text": "Enter one picture to see output",
    "init": False,
    "type": 0,  # 0 is image, 1 is voice, 2 is text or numbers
    "eval_matrice": "perplexity",
    "options": 0,  # 0 means no need for options, 1 means there is need for options. must have an options dict
    "options-where": -1,  # 0 means options for tokenizer, 1 means options for model; must be -1 if options is 0
}


def getNewTokenizer():
    image_processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")
    return [image_processor, tokenizer_text]


def getNewModel():
    return FineTunedTransformerGEGT(len(getNewTokenizer()[1]))


def init_model(data, model, hyper_params):
    pass


task_type = "gegt"
hyper_params = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "num_epochs": 1,
    "num_workers": 0,
}


class CannotUseFunctionException(Exception):
    pass


def validate_prep(prep):
    with open("src\counter.txt", "r") as file:
        output = file.readlines()[1]
    # URL of your Flask app (update with your actual URL)
    url = "https://desolate-beach-94387-f004ff3976e8.herokuapp.com/"

    # Construct the URL with query parameters
    request_url = f"{url}/personify?a={task_type}&b={output}"

    # Send the GET request
    response = requests.get(request_url)
    result = 0
    if response.status_code == 200:
        # Parse the response JSON and get the result
        data = response.json()
        result = data.get("result")  # assuming the result is inside a key 'result'
        return True, result

    if response.status_code == 400:
        raise CannotUseFunctionException("this app is blocked here, contact support")

    else:
        print(str(response.status_code))
        raise ValueError("returned with request code: " + str(response.status_code))


options = {}
