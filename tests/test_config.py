from src.config import LLMConfig


def test_config_defaults():
    config = LLMConfig()
    assert config.batch_size == 4
    assert config.embedding_dim == 64


def test_attention_dimensions():
    config = LLMConfig()
    assert config.embedding_dim % config.num_heads == 0
    