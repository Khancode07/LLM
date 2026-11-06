from src.dataset import TextDataset


def test_next_token_pairs():
    data = TextDataset([1, 2, 3, 4, 5], 3)
    x, y = data[0]
    assert x == [1, 2, 3]
    assert y == [2, 3, 4]


def test_dataset_length():
    data = TextDataset([1, 2, 3, 4, 5], 3)
    assert len(data) == 2