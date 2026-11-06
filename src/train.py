from pathlib import Path
import torch
import torch.nn.functional as F

from src.config import LLMConfig
from src.tokenizer import CharTokenizer, load_text
from src.dataset import TextDataset
from src.model import MiniLLM


def main():
    config = LLMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(__file__).parent.parent / "data" / "raw" / "train.txt"
    text = load_text(path)

    tokenizer = CharTokenizer(text)
    tokens = tokenizer.encode(text)
    dataset = TextDataset(tokens, config.context_length)

    if len(dataset) == 0:
        raise ValueError("Training text is too short!")

    model = MiniLLM(config, len(tokenizer.chars)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate
    )

    print("Device:", device)
    print("Training sequences:", len(dataset))
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    model.train()

    for step in range(1000):
        indices = torch.randint(0, len(dataset), (config.batch_size,))
        batch = [dataset[i.item()] for i in indices]

        x = torch.tensor([a for a, b in batch], dtype=torch.long, device=device)
        y = torch.tensor([b for a, b in batch], dtype=torch.long, device=device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/1000 | Loss: {loss.item():.4f}")

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    tokenizer.save(checkpoint_dir / "tokenizer.json")

    print("Training complete!")
    print("Model saved to checkpoints/model.pt")


if __name__ == "__main__":
    main()