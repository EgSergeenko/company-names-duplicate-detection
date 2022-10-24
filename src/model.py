"""Model architecture."""
import torch


class Model(torch.nn.Module):
    """Class that defines the architecture of the model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        embedding_size: int,
        device: torch.device,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        dropout_fraction: float = 0.2,
    ) -> None:
        """Init model instance.

        Args:
            input_size: Input dimension.
            hidden_size: Hidden dimension.
            embedding_size: Output (embedding) dimension.
            device: Device.
            lstm_num_layers: Number of LSTM layers.
            bidirectional: Is LSTM bidirectional or not.
            dropout_fraction: Dropout probability.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = device
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.dropout_fraction = dropout_fraction
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.dropout_fraction,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(p=self.dropout_fraction)
        self.relu = torch.nn.ReLU()
        fc1_in_features = self.hidden_size
        if self.bidirectional:
            fc1_in_features *= 2
        self.fc1 = torch.nn.Linear(
            in_features=fc1_in_features,
            out_features=self.embedding_size * 2,
        )
        self.fc2 = torch.nn.Linear(
            in_features=self.embedding_size * 2,
            out_features=self.embedding_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward pass.

        Args:
            x: Input for forward pass.

        Returns:
            Output of forward pass.
        """
        h_0, c_0 = self._init_hidden()
        x, (_, _) = self.lstm(x, (h_0, c_0))
        x = self.fc1(x[-1])
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

    def _init_hidden(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.lstm_num_layers
        if self.bidirectional:
            d *= 2
        return (
            torch.zeros(d, self.hidden_size).to(self.device),
            torch.zeros(d, self.hidden_size).to(self.device),
        )
