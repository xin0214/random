import argparse
import json
from sklearn.manifold import TSNE
import os
import pandas as pd
import numpy as np
import torch
from attrdict import AttrDict
import matplotlib.pyplot as plt
from seaborn import clustermap
from sklearn.cluster import SpectralClustering
from transformers import BertConfig
from model import BertForMultiLabelClassification, BertMultiLevelClassifier, BertForMultiLabelClassificationCPCC
from data_loader import GoEmotionsProcessor
import warnings
warnings.filterwarnings('ignore')

""" 
####### Original Code ######## 

*** Idea for label embedding ***

When we perform classification, we will pass the hidden representation through one last linear layer before calculating the loss.
We would like to interpret this last linear layer as the embedding layer of the labels. We explore this idea using some clusterings 
and visualizations here.  

The results provide support to this idea. One approach to inject prior knowledge of label similarity into the learning procedure 
is to impose regularizations on this final linear layer. 
"""

def visualize_last_weight(cli_args):
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    is_all = cli_args.taxonomy == "all"
    processor = GoEmotionsProcessor(args)
    label_list = processor.get_labels()
    if not is_all:
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        if args.use_cpcc == False:
            model = BertForMultiLabelClassification(config=config)
        else:
            model = BertForMultiLabelClassificationCPCC(config=config)
    else:
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            finetuning_task=args.task,
        )
        model = BertMultiLevelClassifier(config=config)

    output_dir = os.path.join(args.output_dir, "current_best")
    filepath = os.path.join(output_dir, "model.pt")
    saved = torch.load(filepath)
    model.load_state_dict(saved['model'])
    if not is_all:
        classifier_weights = model.classifier.weight.data.numpy()
    else:
        classifier_weights = model.classifier_original.weight.data.numpy()
    return classifier_weights, label_list

def get_tsn_plot(label_embed, labels, filepath='layer_embed.png'):
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    layer_embed_tsn = tsne.fit_transform(classifier_weights)
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        x, y = layer_embed_tsn[i, :]
        plt.scatter(x, y, color="blue")
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
    plt.savefig(filepath)

def get_cluster_plot(dot_sim_df):
    clustermap(dot_sim_df)
    plt.savefig("cluster.png")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--taxonomy", type=str, default="original", help="Taxonomy (original, ekman, group, all)")
    cli_parser.add_argument("--ncomponents", type=int, default=6, help="Number of Clusters (2-27)")
    cli_parser.add_argument("--get_plot", type=bool, default=False, help="Whether to save a TSNE plot.")
    cli_args = cli_parser.parse_args()
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    output_dir = os.path.join(args.output_dir, "visualization")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    classifier_weights, label_list = visualize_last_weight(cli_args)
    if args.get_plot:
        get_tsn_plot(classifier_weights, label_list, os.path.join(output_dir, "layer_embed_tsn.png"))
    dot_sim = classifier_weights @ classifier_weights.transpose()
    # Dot product similarities between pairs of labels
    dot_sim_df = pd.DataFrame(dot_sim, columns=label_list, index=label_list)
    n_components = cli_args.ncomponents
    scs = SpectralClustering(n_components).fit(dot_sim_df)
    for c in range(n_components):
        print([label for i, label in enumerate(label_list) if scs.labels_[i] == c])