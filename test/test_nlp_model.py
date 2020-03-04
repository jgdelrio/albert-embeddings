import torch
from transformers import AlbertModel, AlbertTokenizer
import pytest
from hamcrest import assert_that, equal_to, isinstanceof
from albert_emb.utils import get_logger
from albert_emb.nlp_model import get_embeddings, load_model
from albert_emb.config import MODELS, ROOT


logger = get_logger(name='test-albert')

GET_EMBEDDINGS_TEST = [
    ("Hello, world!", ('hello_world.pt', 2, 13), False, "mean"),
    ("Colon discovered America", ('simple_phrase.pt', 3, 24), False, "mean"),
    ("Colon discovered America. Later he returned to Spain.", ('phrase2.pt', 8, 53), False, "mean"),
    (["Colon again.", "He come and go.", "With three carabellas."], ('paragraphs3.pt', 9, 51), False, "mean"),
    ("Colon again. He come and go. With three carabellas.", ('paragraphs3.pt', 9, 51), False, "mean"),
    ("Colon discovered America", ('simple_phrase_sum.pt', 3, 24), False, "sum"),
    ("Some today's news include Macron bid for a tough", ('news.pt', 9, 48), False, "sum"),
]


@pytest.mark.parametrize("text_in, expected, curate, aggregate", GET_EMBEDDINGS_TEST)
def test_get_embeddings(text_in, expected, curate, aggregate):
    sample_ref, exp_word_count, exp_char_count = expected
    exp_embeddings = torch.load(ROOT.joinpath('test', 'samples', sample_ref))

    result = get_embeddings(text_in, curate, aggregate)
    embeddings = result["embeddings"]
    word_count = result["word_count"]
    char_count = result["char_count"]

    assert_that(embeddings.shape[0], equal_to(MODELS['albert-base-v2']['hidden_size']))
    logger.debug(f"Result shape is: {embeddings.shape}")
    assert(torch.all(embeddings.eq(exp_embeddings)))
    assert_that(word_count, equal_to(exp_word_count))
    assert_that(char_count, equal_to(exp_char_count))
    logger.debug("Embeddings value of phrase are correct.")


def test_get_embeddings_curate_type_error():
    with pytest.raises(Exception):
        assert get_embeddings("test", 1)


def test_get_embeddings_aggregate_type_error():
    with pytest.raises(Exception):
        assert get_embeddings("test", False, 42)


def test_get_embeddings_text_type_error():
    with pytest.raises(Exception):
        assert get_embeddings(3.14, False)


def test_load_model_raise_name_error():
    with pytest.raises(Exception):
        assert load_model("non_existing_model")


def test_load_model_albert():
    name = 'albert-base-v2'
    model, tokenizer = load_model(name)

    assert(isinstance(model, AlbertModel))
    assert(isinstance(tokenizer, AlbertTokenizer))


if __name__ == "__main__":
    test_get_embeddings(*GET_EMBEDDINGS_TEST[0])
