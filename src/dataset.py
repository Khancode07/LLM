class TextDataset:
    def __init__(self, token_ids, context_length):
        self.tokens = token_ids
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.tokens) - self.context_length)

    def __getitem__(self, index):
        start = index
        end = start + self.context_length
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y