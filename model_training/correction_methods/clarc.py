import os

import numpy as np
import torch
import tqdm
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from model_training.correction_methods.base_correction_method import LitClassifier, Freeze


def get_layer_activation(model, data, name):
    global layer_act

    def get_layer_act_hook(m, i, o):
        global layer_act
        layer_act = o
        return None

    for layer_name, module in model.named_modules():
        if name == layer_name:
            h = module.register_forward_hook(get_layer_act_hook)

    _ = model(data)
    h.remove()
    return layer_act


def get_features(model, layer_name, dataset, sample_ids, device, batch_size=8):
    subset = dataset.get_subset_by_idxs(sample_ids)
    dl = DataLoader(subset, batch_size=batch_size, shuffle=False)
    features = []
    for data, _ in tqdm.tqdm(dl):
        features.append(get_layer_activation(model, data.to(device), layer_name).detach().cpu())
    return torch.cat(features)


def get_cav(cav_type, model, layer_name, dataset, artifact_name, class_name=None, cav_dir=None):
    basename_cav = f"{cav_dir}/{layer_name}_{artifact_name}"

    fname_cav = f"{basename_cav}_{cav_type}_cav.npy"
    fname_art_mean = f"{basename_cav}_{cav_type}_art_mean.npy"
    fname_no_art_mean = f"{basename_cav}_{cav_type}_no_art_mean.npy"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if (basename_cav is not None) and (os.path.isfile(fname_cav)) and (os.path.isfile(fname_art_mean)) and (
            os.path.isfile(fname_no_art_mean)):
        cav = torch.from_numpy(np.load(fname_cav)).to(device)
        art_mean = torch.from_numpy(np.load(fname_art_mean)).to(device)
        no_art_mean = torch.from_numpy(np.load(fname_no_art_mean)).to(device)
        print(f"Loaded CAV from {fname_cav}")
    else:
        print(f"CAV does not exist ({fname_cav}). Computing now ...")

        assert artifact_name in dataset.sample_ids_by_artifact.keys(), f"Unknown artifact: {artifact_name}, \
            pick one of: {list(dataset.sample_ids_by_artifact.keys())}"

        artifact_sample_ids = dataset.sample_ids_by_artifact[artifact_name]
        if len(artifact_sample_ids) > 1000:
            artifact_sample_ids = np.random.choice(artifact_sample_ids, 1000, replace=False)

        class_id = None if class_name is None else dataset.class_names.index(class_name)

        non_artifact_sample_ids = [i for i in range(len(dataset)) if (i not in artifact_sample_ids) and
                                   ((class_id is None) or (dataset.ids[dataset.class_names[class_id]].values[i] == 1))]

        num_non_artifact_samples = min(len(artifact_sample_ids) * 5, len(non_artifact_sample_ids))
        non_artifact_sample_ids = np.random.choice(np.array(non_artifact_sample_ids), size=num_non_artifact_samples,
                                                   replace=False)

        features_artifact = get_features(model, layer_name, dataset, artifact_sample_ids, device)
        features_non_artifact = get_features(model, layer_name, dataset, non_artifact_sample_ids, device)

        orig_feature_shape = features_artifact.shape

        print(
            f"Computing CAV with {len(features_artifact)} artifact samples ({features_artifact.shape}) and {len(features_non_artifact)} non-artifact samples ({features_non_artifact.shape}).")
        art_mean = features_artifact.mean(dim=0)
        no_art_mean = features_non_artifact.mean(dim=0)

        features = torch.cat([features_artifact, features_non_artifact])
        targets = torch.cat([torch.ones(len(features_artifact)), torch.zeros(len(features_non_artifact))])

        final_shape = features.shape

        if cav_type == "svm":
            ## SVM CAV
            clf = LinearSVC(random_state=0, fit_intercept=False)
            clf.fit(features.reshape(final_shape[0], -1).numpy(), targets.numpy())
            cav = clf.coef_[0].copy()

        elif cav_type == "signal":
            ## Signal CAV
            X = features.reshape(final_shape[0], -1).numpy()
            y = targets.numpy()
            mean_y = y.mean()
            X_residuals = X - X.mean(axis=0)
            covar = (X_residuals * (y - mean_y)[:, np.newaxis]).sum(axis=0) / (y.shape[0] - 1)
            vary = np.sum((y - mean_y) ** 2, axis=0) / (y.shape[0] - 1)
            cav = covar / vary

        else:
            raise ValueError(f"Unknown cav type: {cav_type}")

        cav /= np.linalg.norm(cav)
        cav = torch.tensor(cav.reshape(orig_feature_shape[1:])).to(device)

        np.save(fname_cav, cav.cpu().numpy())
        np.save(fname_art_mean, art_mean.cpu().numpy())
        np.save(fname_no_art_mean, no_art_mean.cpu().numpy())

    return cav, art_mean, no_art_mean


class ClarcFullFeature(LitClassifier):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.initialized = False
        self.model = model
        self.layer_name = config['layer_name']
        self.cav_type = config.get("cav_type", "svm")

        assert "dataset" in kwargs.keys(), "dataset required for ClArCFullFeature"
        assert "class_name" in kwargs.keys(), "class_name required for ClArCFullFeature"
        assert "cav_dir" in kwargs.keys(), "cav_dir required for ClArCFullFeature"
        assert "artifact_name" in kwargs.keys(), "artifact_name has to be passed to ClArCFullFeature"

        self.init_clarc(kwargs['dataset'], kwargs['artifact_name'], kwargs['class_name'], kwargs['cav_dir'])

    def init_clarc(self, dataset, artifact_name, class_name, cav_dir):

        # Compute CAV
        self.cav, self.art_mean, self.no_art_mean = get_cav(self.cav_type, self.model, self.layer_name,
                                                            dataset, artifact_name, class_name, cav_dir)
        self.z = self.get_z()
        self.initialized = True

        # Add ClArC Hook
        for n, m in self.model.named_modules():
            if n == self.layer_name:
                print("Registered forward hook.")
                m.register_forward_hook(self.clarc_hook)

    def clarc_hook(self, m, i, o):
        cav = self.cav.type(o.type())
        z = self.z.type(o.type())

        part_x = torch.matmul(torch.flatten(o, start_dim=1), torch.flatten(cav))
        part_z = torch.matmul(torch.flatten(cav), torch.flatten(z))

        # x' = x + v * (-vTx + vTz) [= x * (I - vvT) + vvTz]
        correction_flat = torch.matmul((part_z - part_x).unsqueeze(1), torch.flatten(cav).unsqueeze(0))
        o_projected = o + correction_flat.reshape(o.shape)

        return o_projected

    def configure_callbacks(self):
        return [Freeze()]
        # return [Freeze(self.layer_name)]


class AClarcFullFeature(ClarcFullFeature):

    def get_z(self):
        return self.art_mean


class PClarcFullFeature(ClarcFullFeature):

    def get_z(self):
        return self.no_art_mean


class Clarc(LitClassifier):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)

        self.layer_name = config["layer_name"]

        self.dataset_name = config["dataset_name"]
        self.model_name = config["model_name"]

        assert "artifact_sample_ids" in kwargs.keys(), "artifact_sample_ids have to be passed to ClArC correction methods"
        assert "n_classes" in kwargs.keys(), "n_classes has to be passed to ClArC correction methods"
        assert "mode" in kwargs.keys(), "mode has to be passed to ClArC correction methods"

        self.artifact_sample_ids = kwargs["artifact_sample_ids"]
        self.n_classes = kwargs["n_classes"]
        self.mode = kwargs['mode']

        print(f"Using {len(self.artifact_sample_ids)} artifact samples.")

        self.path = f"results/global_relevances_and_activations/{self.dataset_name}/{self.model_name}"

        cav, mean_length, mean_length_targets = self.compute_cav(self.mode)
        self.cav = cav
        self.mean_length = mean_length
        self.mean_length_targets = mean_length_targets

        hooks = []
        for n, m in self.model.named_modules():
            if n == self.layer_name:
                print("Registered forward hook.")
                hooks.append(m.register_forward_hook(self.clarc_hook))
        self.hooks = hooks

    def compute_cav(self, mode):
        vecs = []
        sample_ids = []

        path = self.path
        for class_id in range(self.n_classes):
            data = torch.load(f"{path}/{self.layer_name}_class_{class_id}_all.pth")
            if data['samples']:
                sample_ids += data['samples']
                vecs.append(torch.stack(data[mode], 0))

        vecs = torch.cat(vecs, 0)

        linear = LinearSVC(random_state=0, fit_intercept=False)

        sample_ids = np.array(sample_ids)
        target_ids = np.array(
            [np.argwhere(sample_ids == id_)[0][0] for id_ in self.artifact_sample_ids if
             np.argwhere(sample_ids == id_)])
        targets = np.array([1 * (j in target_ids) for j, x in enumerate(sample_ids)])
        num_targets = (targets == 1).sum()
        num_notargets = (targets == 0).sum()
        weights = (targets == 1) * 1 / num_targets + (targets == 0) * 1 / num_notargets
        weights = weights / weights.max()
        X = vecs.detach().cpu().numpy()

        linear.fit(X, targets, sample_weight=weights)
        # print(linear.score(X, targets, sample_weight=weights))
        w = torch.Tensor(linear.coef_)  # [..., None, None]
        cav = w / torch.sqrt((w ** 2).sum())
        # print(torch.argsort(cav, descending=True))
        mean_length = (vecs[targets == 0] * cav).sum(1).mean(0)
        mean_length_targets = (vecs[targets == 1] * cav).sum(1).mean(0)
        # print(f"Computed CAV. {mean_length:.1f} vs {mean_length_targets:.1f}")

        return cav, mean_length, mean_length_targets

    def clarc_hook(self, m, i, o):
        pass

    def configure_callbacks(self):
        return [Freeze(
            # self.layer_name
        )]


class PClarc(Clarc):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)

        self.path = f"results/global_relevances_and_activations/{os.path.basename(config['config_file'])[:-5]}"
        if os.path.exists(self.path):
            print("Re-computing CAV.")
            cav, mean_length, mean_length_targets = self.compute_cav(self.mode)
            self.cav = cav
            self.mean_length = mean_length
            self.mean_length_targets = mean_length_targets
        else:
            if self.hooks:
                for hook in self.hooks:
                    print("Removed hook. No hook should be active for training.")
                    hook.remove()
                self.hooks = []

    def clarc_hook(self, m, i, o):
        outs = m.forward(i[0])
        cav = self.cav.to(outs)
        length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        diff = (self.mean_length.to(outs) - length)
        acts = outs + diff[:, None, None, None] * cav[..., None, None]
        return acts


class AClarc(Clarc):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.lamb = self.config["lamb"]  # 10

    def clarc_hook(self, m, i, o):
        outs = m.forward(i[0])
        cav = self.cav
        length = (outs.flatten(start_dim=2).max(2).values * cav.to(outs)).sum(1)
        mag = (self.mean_length_targets - length).to(cav)
        acts = outs + ((mag * self.lamb)[:, None, None, None] * self.cav[..., None, None]).to(outs)
        # print((acts.flatten(start_dim=2).max(2).values * cav.to(outs)).sum(1))
        # print(outs[:5, 248, 0, 0], acts[:5, 248, 0, 0])
        return acts
