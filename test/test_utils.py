import pytest
from hamcrest import assert_that, equal_to
from albert_emb.utils import get_logger, paragraphs_join


logger = get_logger(name='test-albert')


def test_paragraphs_join():
    input_paragraphs = ["first paragraph", "Second paragraph. ", "3rd paragraph...", '4th and final.']
    expected = "first paragraph. Second paragraph. 3rd paragraph... . 4th and final."

    result = paragraphs_join(input_paragraphs)
    assert_that(result, equal_to(expected), f"Result:\n{result}\nDiffer from:\n{expected}")


if __name__ == "__main__":
    test_paragraphs_join()
