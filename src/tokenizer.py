import json


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"Unknown characters: {unknown}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            chars = json.load(f)
        return cls("".join(chars))


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError("Training text is empty!")

    return text


if __name__ == "__main__":
    tokenizer = CharTokenizer("hello world")
    ids = tokenizer.encode("hello")
    assert tokenizer.decode(ids) == "hello"
    tokenizer.save("tokenizer.json")
    print("Tokenizer tests passed!")