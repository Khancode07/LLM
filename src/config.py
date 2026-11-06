from dataclasses import dataclass


@dataclass
class LLMConfig:
    project_name: str = "LLMs - Made by Khan"
    seed: int = 42
    batch_size: int = 4
    context_length: int = 64
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    learning_rate: float = 0.001


if __name__ == "__main__":
    config = LLMConfig()
    assert config.embedding_dim % config.num_heads == 0
    print(config)
    print("Config test passed!")