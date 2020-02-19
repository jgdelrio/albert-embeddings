import re
import torch
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer, PreTrainedModel, PreTrainedTokenizer
from albert_emb.config import DEFAULT_MODEL, MODELS, ROOT
from albert_emb.utils import get_logger


SUPPORTED = ["ALBERT"]                      # Models supported
AGGREGATION_MODES = ("mean", "sum")         # Aggregation modes
find_divisions = re.compile(r'[\s+]')       # Find word divisions


def load_model(name=DEFAULT_MODEL):
    if not isinstance(name, str):
        raise TypeError('name must be string')

    if name in MODELS.keys():
        # Initializing an ALBERT-base style configuration
        # albert_params = ALBERT['params']
        # albert_base_configuration = AlbertConfig(
        #     hidden_size=albert_params["hidden_size"],
        #     num_attention_heads=albert_params["num_attention_heads"],
        #     intermediate_size=albert_params["intermediate_size"],
        # )

        model_ref = MODELS[name]

        # Load pre-trained model tokenizer (vocabulary)  ex:   'albert-base-v2', 'albert-xxlarge-v1'
        print("Loading tokenizer")
        tokenizer = AlbertTokenizer.from_pretrained(f"{model_ref['reference']}_tok")
        # Load pre-trained model (weights)
        print("Loading Model")
        model = AlbertModel.from_pretrained(model_ref['reference'])
        print("Load finished")
        return model, tokenizer

    else:
        raise ValueError(f"model {name} is not yet supported. Please use one of: {SUPPORTED}")


def retrieve_model(name=DEFAULT_MODEL, logger=get_logger('model-retrieval')):
    model_folder = ROOT.joinpath('model', name)
    model_folder.mkdir(parents=True, exist_ok=True)
    if model_folder.joinpath('pytorch_model.bin').exists():
        logger.info(f"Model {name}: already available")
    else:
        try:
            model = AlbertModel.from_pretrained(name)
            PreTrainedModel.save_pretrained(model, model_folder)
        except Exception as err:
            logger.error(f"ERROR loading the requested model {name}: {err}")

    tokenizer_folder = ROOT.joinpath('model', f"{name}_tok")
    tokenizer_folder.mkdir(parents=True, exist_ok=True)
    if tokenizer_folder.joinpath('spiece.model').exists():
        logger.info(f"Tokenizer {name}: already available")
    else:
        try:
            tokenizer = AlbertTokenizer.from_pretrained(name)
            PreTrainedTokenizer.save_pretrained(tokenizer, tokenizer_folder)
        except Exception as err:
            logger.error(f"ERROR loading the requested tokenizer {name}: {err}")


# Load model and tokenizer
model, tokenizer = load_model()
# Put the model in 'evaluation' mode (feed-forward operation)
model.eval()

# Accessing the model configuration
configuration = model.config


def get_embeddings(text, curate=True, aggregate="mean", logger=get_logger()):
    if not isinstance(curate, bool):
        raise TypeError('curate must be boolean')
    if not isinstance(aggregate, str):
        raise TypeError('aggregate must be a valid string')
    elif aggregate not in AGGREGATION_MODES:
        raise ValueError(f"aggregate '{aggregate}' must be a valid strategy: {AGGREGATION_MODES}")
    if isinstance(text, (list, tuple)):
        # Received list of paragraphs
        if any([not isinstance(k, str) for k in text]):
            raise TypeError("all elements within the list must be string type")
        # Join into single text with new lines
        word_count = sum(map(lambda x: len(x.split(" ")), text))
        text = "\n".join(text)
        char_count = len(text)

    elif isinstance(text, str):
        word_count = len(find_divisions.findall(text)) + 1
        char_count = len(text)
    else:
        raise TypeError("text must be a string")

    if curate:
        logger.debug(f"Curating text: {text}")
        # TODO
        # text = curated_text(text)

    # Encode text (mark and tokenize)
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.

    # Deactivate the gradient calculations (saving memory and speeding computation)
    # We don't require grads or backpropagation since we just do the forward pass
    with torch.no_grad():
        # Get hidden states features for each layer
        hidden_states, _ = model(input_ids)             # Models outputs are tuples

    logger.debug(f"Size of the tensor: {hidden_states.size()}")
    # Squeeze batched dimension
    shape = len(hidden_states.size())
    if shape == 4:
        # Only useful in models with different layer parameters
        token_embeddings = torch.squeeze(hidden_states, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
    elif shape == 3:
        token_embeddings = torch.squeeze(hidden_states, dim=0)
    else:
        raise ValueError(f"Unsupported number of dimensions: {shape}")

    # Return a 1 dim tensor if aggregate is requested
    if aggregate == "mean":
        logger.debug(f"Aggregating tokens by mean...")
        embeddings = torch.mean(token_embeddings, dim=0)
    elif aggregate == "sum":
        logger.debug(f"Aggregating tokens by sum...")
        embeddings = torch.sum(token_embeddings, dim=0)
    else:
        raise ValueError(f"Unsupported strategy: {aggregate}")

    return {
        "embeddings": embeddings,
        "word_count": word_count,
        "char_count": char_count,
        "model_name": MODELS[DEFAULT_MODEL]['name'],
        "model_version": MODELS[DEFAULT_MODEL]['version']}
