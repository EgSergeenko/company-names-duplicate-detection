"""Data utils for model usage."""
import torch
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """Dataset for company names."""

    def __init__(
        self,
        sequences: list[str],
        labels: list[int],
        symbols: str,
    ) -> None:
        """Init dataset instance.

        Args:
            sequences: List of strings.
            labels: List of sequence labels.
            symbols: List of allowed symbols.
        """
        self.symbols = symbols
        self.symbol_ids = {
            symbol: idx for idx, symbol in enumerate(self.symbols)
        }
        self.label_ids = {
            label: idx for idx, label in enumerate(set(labels))
        }
        self.encodings = [
            self._get_sequence_encoding(sequence) for sequence in sequences
        ]
        self.labels = [
            self._get_label_encoding(label) for label in labels
        ]

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            Dataset length.
        """
        return len(self.encodings)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample by index.

        Args:
            idx: Index of sample.

        Returns:
            A pair of encoding and label.
        """
        return self.encodings[idx], self.labels[idx]

    def _get_symbol_encoding(self, symbol: str) -> torch.Tensor:
        encoding = torch.zeros(len(self.symbols))
        encoding[self.symbol_ids[symbol]] = 1.0
        return encoding

    def _get_sequence_encoding(self, sequence: str) -> torch.Tensor:
        return torch.stack(
            [self._get_symbol_encoding(symbol) for symbol in sequence],
        )

    def _get_label_encoding(self, label: int) -> torch.Tensor:
        return torch.tensor(self.label_ids[label], dtype=torch.long)
