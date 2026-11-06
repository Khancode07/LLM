import torch
from src.config import LLMConfig
from src.model import MiniLLM


def test_model_output_shape():
    config = LLMConfig()
    model = MiniLLM(config, vocab_size=30)

    x = torch.randint(0, 30, (2, config.context_length))
    output = model(x)

    assert output.shape == (2, config.context_length, 30)


def test_model_has_trainable_parameters():
    model = MiniLLM(LLMConfig(), vocab_size=30)

    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0