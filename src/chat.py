from pathlib import Path
import torch

from src.config import LLMConfig
from src.model import MiniLLM
from src.tokenizer import CharTokenizer


def main():
    root = Path(__file__).parent.parent
    folder = root / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharTokenizer.load(folder / "tokenizer.json")
    config = LLMConfig()
    model = MiniLLM(config, len(tokenizer.chars)).to(device)

    model.load_state_dict(
        torch.load(folder / "model.pt", map_location=device, weights_only=True)
    )
    model.eval()

    prompt = input("Tum: ")
    try:
        ids = tokenizer.encode(prompt)
    except ValueError as error:
        print(error)
        return

    if not ids:
        ids = [0]

    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(100):
            context = x[:, -config.context_length:]
            logits = model(context)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_id], dim=1)

    print("Khan LLM:", tokenizer.decode(x[0].tolist()))


if __name__ == "__main__":
    main()