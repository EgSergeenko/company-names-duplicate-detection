"""Logic for training models."""
import logging
import os

import hydra
import torch
import tqdm
from omegaconf import DictConfig
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from common import (evaluate_f1_score, get_data_split, get_embeddings,
                    get_logger, load_data, set_seed)
from data import Dataset
from model import Model


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def train(config: DictConfig) -> None:
    """Train model.

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

    train_data, val_data, _ = get_data_split(
        sequences,
        labels,
        augmentation,
        config.val_size,
        config.test_size,
    )
    train_sequences, train_labels = train_data[0], train_data[1]
    val_sequences, val_labels = val_data[0], val_data[1]

    train_dataset = Dataset(
        sequences=train_sequences,
        labels=train_labels,
        symbols=config.symbols,
    )
    val_dataset = Dataset(
        sequences=val_sequences,
        labels=val_labels,
        symbols=config.symbols,
    )
    sampler = MPerClassSampler(
        labels=train_labels,
        m=config.sampler.m,
        batch_size=config.train_loader.batch_size,
        length_before_new_iter=len(train_dataset),
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=lambda batch: batch,
        batch_size=config.train_loader.batch_size,
        sampler=sampler,
        shuffle=config.train_loader.shuffle,
        drop_last=config.train_loader.drop_last,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        collate_fn=lambda batch: batch,
        batch_size=config.val_loader.batch_size,
        shuffle=config.val_loader.shuffle,
        drop_last=config.val_loader.drop_last,
    )

    model = Model(
        input_size=len(config.symbols),
        hidden_size=config.model.hidden_size,
        embedding_size=config.model.embedding_size,
        device=device,
    ).to(device)
    criterion = ArcFaceLoss(
        num_classes=len(set(train_labels)),
        embedding_size=config.model.embedding_size,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        config.optimizer.lr,
    )
    criterion_optimizer = AdamW(
        criterion.parameters(),
        config.criterion_optimizer.lr,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=config.scheduler.milestones,
        gamma=config.scheduler.gamma,
    )

    best_score, best_threshold = 0.0, 0.0
    for epoch in range(config.epochs):
        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            criterion_optimizer,
            device,
            epoch + 1,
        )
        score, threshold = eval_epoch(
            model,
            val_loader,
            device,
            epoch + 1,
            logger,
        )
        if score > best_score:
            best_score = score
            best_threshold = threshold
            save_model(
                model,
                config.checkpoint.save.dir,
                config.checkpoint.save.filename,
            )
        scheduler.step()
    logger.info(
        'Best F1-score: {0:.5f}, Best threshold: {1}'.format(
            best_score,
            best_threshold,
        ),
    )


def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    criterion_optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> None:
    """Train one epoch.

    Args:
        model: Model to train.
        data_loader: Data to train.
        optimizer: Optimizer for model.
        criterion: Loss function.
        criterion_optimizer: Optimizer for loss function.
        device: Device to use.
        epoch: Epoch number.
    """
    description = 'Training... Epoch: {0}'.format(
        epoch,
    )
    model.train()
    progress_bar = tqdm.tqdm(data_loader, desc=description)
    for batch in progress_bar:
        batch_embeddings, batch_labels = [], []
        for sample in batch:
            encoding, label = sample
            output = model(encoding.to(device))
            batch_embeddings.append(output)
            batch_labels.append(label)

        embeddings = torch.stack(batch_embeddings).to(device)
        labels = torch.stack(batch_labels).to(device)

        optimizer.zero_grad()
        criterion_optimizer.zero_grad()
        loss = criterion(embeddings, labels)
        loss.backward()
        description = 'Training... Epoch: {0}, Loss: {1:.5f}'.format(
            epoch,
            loss.item(),
        )
        progress_bar.set_description(description)
        optimizer.step()
        criterion_optimizer.step()


def eval_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
) -> tuple[float, float]:
    """Evaluate one epoch.

    Args:
        model: Model to evaluate.
        data_loader: Data to evaluate.
        device: Device to use.
        epoch: Epoch number:
        logger: Logger object.

    Returns:
        The best f1 score for epoch with the best threshold.
    """
    logger.info(
        'Evaluating... Epoch: {0}'.format(
            epoch,
        ),
    )
    embeddings, labels = get_embeddings(model, data_loader, device)
    scores = evaluate_f1_score(
        embeddings,
        labels,
    )
    best_score, best_threshold = 0.0, 0.0
    for score, threshold in scores:
        if score > best_score:
            best_score = score
            best_threshold = threshold
    logger.info(
        'Epoch: {0}, F1-score: {1:.5f}, Threshold: {2:.7f}'.format(
            epoch,
            best_score,
            best_threshold,
        ),
    )
    return best_score, best_threshold


def save_model(
    model: torch.nn.Module,
    checkpoint_dir: str,
    checkpoint_filename: str,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model weights.
        checkpoint_dir: Path to directory with file.
        checkpoint_filename: Filename.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(
        model,
        os.path.join(
            checkpoint_dir,
            checkpoint_filename,
        ),
    )


if __name__ == '__main__':
    train()
