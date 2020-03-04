from os import getenv
import pathlib


# API Info
API_VERSION = "0.0.1"

# Parameters
LOG_LEVEL = getenv("LOG_LEVEL", "INFO")
ROOT = pathlib.Path(__file__).parents[1]
MODEL_FOLDER = ROOT.joinpath('model')
PORT = getenv("PORT", "5000")

# Model Info & Parameters
DEFAULT_MODEL = getenv("DEFAULT_MODEL", "albert-base-v2")
MODELS = {
    'albert-base-v2': {
        "name": "ALBERT-Base",
        "version": "2.0",
        "reference": MODEL_FOLDER.joinpath('albert-base-v2').as_posix(),
        "hidden_size": 768,
    },
    'albert-large-v1': {
        "name": "ALBERT-Large",
        "version": "1.0",
        "reference": MODEL_FOLDER.joinpath('albert-large-v1').as_posix(),
        "hidden_size": 768,
    },
    'albert-xxlarge-v1': {
        "name": "ALBERT-xxLarge",
        "version": "1.0",
        "reference": MODEL_FOLDER.joinpath('albert-xxlarge-v1').as_posix(),
        "hidden_size": 768,
    },
}

# Other possible models include:
#       (BertModel,       BertTokenizer,       'bert-base-uncased')
#       (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt')
#       (GPT2Model,       GPT2Tokenizer,       'gpt2')
#       (CTRLModel,       CTRLTokenizer,       'ctrl')
#       (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103')
#       (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased')
#       (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024')
#       (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased')
#       (RobertaModel,    RobertaTokenizer,    'roberta-base')
#       (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base')
