import torch
import tqdm

from utils.metrics import get_accuracy, get_auc, get_f1


def compute_model_scores(
        model: torch.nn.Module,
        dl: torch.utils.data.DataLoader,
        device: str):
    model.to(device).eval()
    model_outs = []
    ys = []
    for x_batch, y_batch in tqdm.tqdm(dl):
        model_out = model(x_batch.to(device)).detach().cpu()
        model_outs.append(model_out)
        ys.append(y_batch)

    model_outs = torch.cat(model_outs)
    y_true = torch.cat(ys)

    return model_outs, y_true


def compute_metrics(model_outs, y_true, class_names=None, prefix="", suffix=""):
    results = {
        f"{prefix}auc{suffix}": get_auc(model_outs, y_true),
        f"{prefix}accuracy{suffix}": get_accuracy(model_outs, y_true),
        f"{prefix}f1{suffix}": get_f1(model_outs, y_true)
    }

    return results
