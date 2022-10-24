"""Logic for testing performance of models."""
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from common import (evaluate_f1_score, get_data_split, get_embeddings,
                    get_logger, load_data, set_seed)
from data import Dataset


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def test(config: DictConfig) -> None:
    """Test model performance.

    Args:
        config: Experiment settings.
    """
    logger = get_logger()
    device = torch.device(config.device)
    set_seed(config.seed)

    sequences, labels, augmentation = load_data(
        config.data.dir,
        config.data.filename,
    )

    _, _, test_data = get_data_split(
        sequences,
        labels,
        augmentation,
        config.val_size,
        config.test_size,
    )
    test_sequences, test_labels = test_data[0], test_data[1]

    test_dataset = Dataset(
        sequences=test_sequences,
        labels=test_labels,
        symbols=config.symbols,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=lambda batch: batch,
        batch_size=config.val_loader.batch_size,
        shuffle=config.val_loader.shuffle,
        drop_last=config.val_loader.drop_last,
    )

    model = load_model(
        config.checkpoint.load.dir,
        config.checkpoint.load.filename,
        device,
    )

    embeddings, labels = get_embeddings(model, test_loader, device)
    scores = evaluate_f1_score(embeddings, test_labels, config.threshold)
    logger.info(
        'F1-score: {0:.5f}, Threshold: {1}'.format(
            scores[0][0],
            config.threshold,
        ),
    )


def load_model(
    checkpoint_dir: str,
    checkpoint_filename: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load model weights.

    Args:
        checkpoint_dir: Path to directory with file
        checkpoint_filename: Filename.
        device: Device to map.

    Returns:
        Loaded model.
    """
    return torch.load(
        os.path.join(
            checkpoint_dir,
            checkpoint_filename,
        ),
        map_location=device,
    )


if __name__ == '__main__':
    test()
