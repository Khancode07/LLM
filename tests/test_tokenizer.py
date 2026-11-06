import pytest
from src.tokenizer import CharTokenizer


def test_encode_decode():
    t = CharTokenizer("hello world")
    assert t.decode(t.encode("hello")) == "hello"


def test_token_ids():
    t = CharTokenizer("abc")
    assert t.encode("cab") == [2, 0, 1]


def test_unknown_character():
    t = CharTokenizer("abc")
    with pytest.raises(ValueError):
        t.encode("z")


def test_save_load():
    t = CharTokenizer("hello world")
    t.save("test_tokenizer.json")
    loaded = CharTokenizer.load("test_tokenizer.json")
    assert loaded.decode(loaded.encode("hello")) == "hello"