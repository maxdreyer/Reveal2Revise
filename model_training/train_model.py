import tqdm
import torch
import gc
import logging
import wandb
import os
from utils.metrics import get_auc_label, get_auc

logger = logging.getLogger(__name__)

def log_results(results, do_wandb_logging, e):
    if do_wandb_logging:
        wandb.log(results, step=e, commit=True)
    logger.info(f"Epoch {e}: {results}")

def store_model(model, savename):
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    torch.save(model.state_dict(), savename)

def train_model(
    model: torch.nn.Module, 
    model_name: str,
    dl_train: torch.utils.data.DataLoader,
    dl_val_dict: dict,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    eval_every: int,
    store_every: int,
    device: str,
    model_savedir: str,
    do_wandb_logging: bool
    ):
    """
    Train skin cancer recognition model for given parameters.

    Args:
        model (torch.nn.Module): Model to be trained
        model_name (str): name of model type
        dl_train (torch.utils.data.DataLoader): DataLoader for training data
        dl_val (torch.utils.data.DataLoader): DataLoader for validation data
        criterion (torch.nn.Module): Loss function to be optimized
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of training epochs
        eval_every (int): Evaluate model with validation data every n epochs.
        store_every (int): Store model weights every n epochs.
        device (str): Device to be trained on ('cuda'/'cpu')
        model_savedir (str): Directory where model is stored.
        do_wandb_logging (bool): boolean specifying whether results are logged to weights and biases
    """

    for epoch in range(1, num_epochs+1):
        metrics_epoch = run_one_epoch(model, dl_train, criterion, optimizer, device, update_params=True)
        metrics_epoch = {f"train_{key}": val for key, val in metrics_epoch.items()}

        if epoch % eval_every == 0:
            for val_name, dl_val in dl_val_dict.items():
                logger.info(f"Running evaluation for {val_name}")
                metrics_val = run_one_epoch(model, dl_val, criterion, optimizer, device, update_params=False)
                metrics_val = {f"{val_name}_{key}": val for key, val in metrics_val.items()}
                metrics_epoch = {**metrics_epoch, **metrics_val}
        
        log_results(metrics_epoch, do_wandb_logging, epoch)
        
        if epoch % store_every == 0:
            store_model(model, f"{model_savedir}/checkpoint_{model_name}_{epoch}.pth")

        if scheduler:
            scheduler.step()

    store_model(model, f"{model_savedir}/checkpoint_{model_name}_last.pth")

def run_one_epoch(
    model: torch.nn.Module, 
    dl: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    update_params: bool
    ):
    """
    Runs the model for one epoch, either for training or validation.

    Args:
        model (torch.nn.Module): Model to be trained
        dl (torch.utils.data.DataLoader): DataLoader with training/validation data.
        criterion (torch.nn.Module): Loss function to be optimized
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to be trained on ('cuda'/'cpu')
        update_params (bool): Boolean specifying wether models weights are to be updated (training) or not (validation)

    Returns:
        dict: Dictionary with metrics to be logged
    """

    model.to(device)
    running_loss = torch.tensor(0).float()

    model.train() if update_params else model.eval()

    y_true = []
    model_outs = []

    for i, (imgs, labels) in enumerate(tqdm.tqdm(dl)):
        if update_params:
            optimizer.zero_grad()

        imgs = imgs.to(device)
        outputs = model(imgs)
        outputs= outputs.cpu()
        loss = criterion(outputs, labels)

        if update_params:
            loss.backward()
            optimizer.step()
        
        model_outs.append(outputs.detach())
        y_true.append(labels)

        running_loss += loss.data.clone().cpu()

    y_true = torch.cat(y_true)
    model_outs = torch.cat(model_outs)

    y_hat = model_outs.argmax(1)

    results = {
        'loss': running_loss.item() / len(dl.dataset),
        'accuracy': (y_true == y_hat).numpy().mean(),
        'AUC': get_auc(model_outs, y_true)
        }

    results_auc = {f"AUC_{dl.dataset.class_names[class_id]}": get_auc_label(y_true, model_outs, class_id)
                    for class_id in range(model_outs.shape[1])}

    model.cpu()
    torch.cuda.empty_cache(); gc.collect()

    return {**results, **results_auc}

