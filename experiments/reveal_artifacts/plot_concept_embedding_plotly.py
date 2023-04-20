import glob
import json
import multiprocessing
import os
import random
from argparse import ArgumentParser
from typing import List

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import torch
import yaml
from crp.attribution import CondAttribution
from crp.cache import ImageCache
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, load_maximization
from crp.image import vis_opaque_img
from crp.visualization import FeatureVisualization
from dash import Dash, html, dcc, Output, Input, State, callback_context
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

app = Dash(__name__,
           suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([

    html.Div(children=[
        html.H1('Concept Embedding'),
        dbc.Button(id='button', n_clicks=0, children='Run', color="secondary", outline=True,
                   style={"display": "none"}),
        dcc.Loading(children=[
            dcc.Graph(
                id='t-sne',
                responsive=True,
                config={'scrollZoom': True, 'displaylogo': False},
                style={"width": "100%", "height": "75vh"})
        ], type="circle"),
    ]),
    html.H1('Concept Visualizations'),
    dcc.Loading(children=[
        html.Div(id="image_container"),
    ], type="circle"),
], style={"max-width": "1000px", "margin": "0 auto"})


def get_args(fixed_arguments: List[str] = []):
    parser = ArgumentParser()
    parser.add_argument("--layer_name", default="features.26", type=str)
    parser.add_argument('--config_file',
                        default="config_files/correcting_isic/local/vgg16_Vanilla.yaml")

    args = parser.parse_args()

    with open(parser.parse_args().config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args


args = get_args(fixed_arguments=["layer_name"])

dataset = get_dataset(args.dataset_name)(data_paths=args.data_paths,
                                         normalize_data=True,
                                         split="train")

model = get_fn_model_loader(model_name=args.model_name)(n_class=len(dataset.class_names),
                                                        ckpt_path=args.ckpt_path_corrected)
model = model.to("cuda")
model.eval()

canonizers = get_canonizer(args.model_name)
composite = EpsilonPlusFlat(canonizers)
cc = ChannelConcept()

layer_names = get_layer_names(model, [nn.Conv2d])
layer_map = {layer: cc for layer in layer_names}

attribution = CondAttribution(model)

cache = ImageCache()
ds = get_dataset(args.dataset_name)(data_paths=args.data_paths, normalize_data=False, split="train")
fv = FeatureVisualization(attribution, ds, layer_map, path=f"crp_files/{args.config_name}")  # TODO: make cache work


def add_plot(fig, x_, y_, img):
    fig.add_layout_image(
        x=x_,
        y=y_,
        source=img,
        xref="x",
        yref="y",
        sizex=4,
        sizey=4,
        sizing="stretch",
        xanchor="center",
        yanchor="middle",
    )
    return fig['layout']['images'][0]


@app.callback(Output('t-sne', 'figure'), Input('button', 'n_clicks'), )
def main(n_clicks):
    dataloader = DataLoader(dataset.get_subset_by_idxs(dataset.idxs_val),
                            shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=8)
    saving_path = f"results/concept_similarity/{args.config_name}"

    try:
        data = torch.load(f"{saving_path}/similarities.pth")
    except:
        print("Computing concept similarities...")
        compute_similarity(model, dataloader, saving_path)
        data = torch.load(f"{saving_path}/similarities.pth")

    layer_name = args.layer_name
    cos_pos = data["cosine_sim"][layer_name]

    embedding = TSNE(metric='precomputed', perplexity=3, random_state=56, learning_rate=500, init="random")

    X = embedding.fit_transform((1 - cos_pos).abs() / 2)

    x, y = X[:, 0], X[:, 1]
    fig = px.scatter(x=x, y=y, title=None, hover_data={'id': np.arange(len(x))})
    print("Computing reference images...")
    Nth = 1
    # TODO: should be parallelized for speed
    ref_imgs = fv.get_max_reference(range(len(x))[::Nth], layer_name, "relevance", (0, 1), rf=True,
                                    composite=composite, plot_fn=vis_opaque_img)

    imgs = [x[0] for x in ref_imgs.values()]

    print("Plotting...")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    res = [pool.apply_async(add_plot, args=(fig, x_, y_, img)) for x_, y_, img in zip(x[::Nth], y[::Nth], imgs)]
    for r in res:
        r.wait()

    fig_imgs = []
    for r in res:
        fig_imgs.append(r.get())

    pool.close()
    fig['layout']['images'] = [f for f in fig_imgs]

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # set x and y label
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    fig.update_layout(dragmode='select', autosize=False)
    return fig


@app.callback(
    Output('image_container', 'children'),
    Input('t-sne', 'selectedData'))
def display_img(data):
    if data is None:
        return {}
    children = []
    for k, point in enumerate(data["points"]):
        concept_id = point["pointIndex"]
        print("load images for", concept_id)
        ref_imgs = fv.get_max_reference([concept_id], args.layer_name,
                                        "relevance", r_range=(0, 16), composite=composite, rf=True,
                                        plot_fn=vis_opaque_img)[concept_id]

        imgs = [np.array(x.resize((100, 100))) for x in ref_imgs]

        fig = px.imshow(np.array(imgs), facet_col=0, binary_string=True, labels={},
                        facet_col_spacing=0, facet_row_spacing=0, facet_col_wrap=8)

        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_traces(hoverinfo='none', hovertemplate=None)

        for annot in fig.layout.annotations:
            annot['text'] = ''

        child = dcc.Graph(
            responsive=True,
            figure=fig,
            id="images",
            style={"width": "100%", "height": "200px", "padding": "0", "margin": "1em auto", "max-width": "1000px"},
        )
        title = html.H3(f"Concept {concept_id}", style={"margin": "1em auto", "max-width": "1000px"})
        modal = html.Div(
            [
                dbc.Button("Collect Reference Samples", id=f"open_{k}", n_clicks=0, key=concept_id,
                           name=concept_id),
            ]
        )

        children.append(title)
        children.append(modal)
        children.append(child)

    for l in range(k + 1, 20):
        children.append(
            dbc.Button("Collect Reference Samples", id=f"open_{l}", n_clicks=0, key=l, style={"display": "none"}))
        # children.append(dbc.Button("Close", id=f"close_{l}", n_clicks=0, style={"display": "none"}))

    children.append(dcc.Loading(dbc.Modal(
        children=[
            dbc.ModalHeader(dbc.ModalTitle("Header")),
            dcc.Loading(dbc.ModalBody("This is the content of the modal")),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id=f"close", className="ms-auto", n_clicks=0
                )
            ),
        ],
        id="modal",
        is_open=False,
        size="xl",
    )), )
    return children


@app.callback(
    Output("modal", "is_open"),
    *[[Input(f"open_{concept_id}", "n_clicks"), Input(f"open_{concept_id}", "key")] for concept_id in
      range(20)],
    Input(f"close", "n_clicks"),
    State("modal", "is_open"),
)
def toggle_modal(*args):
    trigger = callback_context.triggered[0]
    print("You clicked button {}".format(trigger["prop_id"]))
    is_open = args[-1]
    n2 = args[-2]
    if trigger["prop_id"] != "." and "close" not in trigger["prop_id"]:

        n1 = args[int(trigger["prop_id"].split(".")[0].split("_")[-1]) * 2]
        key = args[int(trigger["prop_id"].split(".")[0].split("_")[-1]) * 2 + 1]
        print(key)
        if n1 or n2:
            return not is_open
    if n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal", "children"),
    *[[Input(f"open_{concept_id}", "n_clicks"), Input(f"open_{concept_id}", "key")] for concept_id in
      range(20)],
    Input("modal", "is_open"),
)
def change_modal(*args_):
    trigger = callback_context.triggered[0]
    print("You clicked button {}".format(trigger["prop_id"]))
    if trigger["prop_id"] != "." and args_[-1]:
        key = args_[int(trigger["prop_id"].split(".")[0].split("_")[-1]) * 2 + 1]

        concept_id = key
        print("load images for", concept_id)
        ref_imgs = fv.get_max_reference([concept_id], args.layer_name,
                                        "relevance", r_range=(0, 80), composite=composite, rf=True,
                                        plot_fn=vis_opaque_img)[concept_id]
        ref_sample_ids = load_maximization(fv.RelMax.PATH, args.layer_name)[0][:, concept_id]
        children = []
        imgs = [np.array(x.resize((100, 100))) for x in ref_imgs]
        for img in np.array(imgs[:]):
            fig = px.imshow(img, labels={}, height=100)
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            fig.update_traces(hoverinfo='none', hovertemplate=None)

            for annot in fig.layout.annotations:
                annot['text'] = ''

            child = (dcc.Graph(
                responsive=True,
                figure=fig,
                style={"width": "125px", "height": "125px", "padding": "0", "margin": "0em 0"},
                config={
                    'displayModeBar': False
                }
            ))
            children.append(child)

        checklist = dcc.Checklist(
            [
                {
                    "label": [
                        child
                    ],
                    "value": dataset.get_sample_name(ref_sample_ids[i])
                }
                for i, child in enumerate(children)
            ],
            id="checklist",
            inline=True,
            value=[dataset.get_sample_name(ref_sample_ids[i]) for i in range(len(children))],
            labelStyle={"display": "flex", "align-items": "center", "justify-content": "center",
                        "flex-direction": "column", "row-gap": ".2em"},
            style={"display": "flex", "flex-wrap": "wrap", "width": "100%",
                   "justify-content": "space-around", "row-gap": "1em"}
        )

        json_files = glob.glob("data/*.json")

        return [
            dbc.ModalHeader(dbc.ModalTitle(f"Collect Reference Samples for Concept {key}")),
            dcc.Loading(dbc.ModalBody(
                children=[html.H3("Please choose the reference images that correspond to an artifact."),
                          html.Div(children=checklist)])),
            dbc.ModalFooter(

                [
                    html.Div([dbc.Label("Select artifact JSON file:"),
                              dbc.Select(
                                  json_files,
                                  json_files[0],
                                  id="json-select",
                              ), ]),
                    html.Div([
                        html.Div([dbc.Label("Artifact Name:"),
                                  dbc.Input(placeholder="Name goes here...", type="text", id="artifact_name"),
                                  dbc.RadioItems(
                                      options=[
                                          {"label": "Replace artifact samples", "value": 1},
                                          {"label": "Add artifact samples", "value": 2},
                                      ],
                                      value=2,
                                      id="switches-input",
                                      style={"margin": "1em 0"}
                                  ),
                                  dbc.Button("Save to JSON", id=f"export", className="ml-2", n_clicks=0),

                                  ], ),

                    ]),
                    dbc.Alert(
                        "Saved successfully!",
                        id="alert-auto",
                        color="success",
                        is_open=False,
                        duration=3000,
                    ),
                    dbc.Button(
                        "Close", id=f"close", className="ms-auto", n_clicks=0
                    ),

                ],
                style={"align-items": "flex-start", }
            ), ]
    else:
        return dcc.Loading([
            dbc.ModalHeader(dbc.ModalTitle("")),
            dbc.ModalBody("Loading..."),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id=f"close", className="ms-auto", n_clicks=0
                )
            ),
        ])


@app.callback(
    [Output("alert-auto", "is_open"), Output("alert-auto", "color"), Output("alert-auto", "children"),
     Output("export", "n_clicks")],
    Input("export", "n_clicks"),
    State("json-select", "value"),
    State("artifact_name", "value"),
    State("switches-input", "value"),
    State("checklist", "value"),
)
def export_json(n_clicks, json_file, artifact_name, switches_input, checklist):
    if n_clicks:
        print("export json")
        print(json_file, artifact_name, switches_input)

        if not artifact_name:
            return [True, "warning", "Artifact name required!", n_clicks]

        if not checklist:
            return [True, "warning", "No samples chosen!", n_clicks]

        with open(json_file, "r") as f:
            data = json.load(f)
        data.setdefault(artifact_name, [])
        if switches_input == 1:
            data[artifact_name] = []
        for i in checklist:
            if i not in data[artifact_name]:
                data[artifact_name].append(i)
        with open(json_file, "w") as f:
            json.dump(data, f, indent=1)
        return [True, "success", "Saved successfully!", n_clicks]

    return [False, "", "", n_clicks]


def compute_similarity(model, dataloader, path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # get layers
    bn_layers = [m for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]

    if not bn_layers:
        layers = [m for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
        layer_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    else:
        layers = bn_layers
        layer_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]

    intermediate_outs = [[] for _ in layers]
    resize = Resize(32)

    def forward_hook(ind):
        def forward(m, inp, o):
            v = m.forward(inp[0]).detach().cpu()
            if v.shape[-1] > 33:
                v = resize(v)
            intermediate_outs[ind].append(v)

        return forward

    forward_hooks = [m.register_forward_hook(forward_hook(i)) for i, m in enumerate(layers)]

    for j, (img, target) in enumerate(tqdm(iter(dataloader))):
        img = img.to(device)
        model(img)

    [forward_hook.remove() for forward_hook in forward_hooks]

    intermediate_outs = [torch.cat(x).transpose(1, 0).flatten(start_dim=1) for x in intermediate_outs]

    print("compute correlation")
    correlation = [(o - o.mean(0)[None]) / (o.std(0)[None] + 1e-8) for o in intermediate_outs]
    correlation = [(o @ o.t() / o.shape[-1]).cpu().abs() for o in correlation]
    print("compute cosine similarity")
    cosine_sim = [cosinesim_matrix(o.clamp(min=0)) for o in intermediate_outs]
    print("compute weight cosine similarity")

    os.makedirs(path, exist_ok=True)
    torch.save(
        {'correlation': dict(zip(layer_names, correlation)), 'cosine_sim': dict(zip(layer_names, cosine_sim))},
        f"{path}/similarities.pth")


def cosinesim_matrix(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(X) @ torch.nn.functional.normalize(X).t()


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=False, port=8051)
