import os
from argparse import ArgumentParser
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def write_confusion_matrix(label_set, ref_labels, pred_labels, outpath, ax):
    label2idx = {label: idx for idx, label in enumerate(label_set)}
    categories = [cat for cat in label_set]
    agreement_table = np.zeros((len(label_set), len(label_set)))
    for ref, pred in zip(ref_labels, pred_labels):
        if pred is None:
            continue
        if ref not in label2idx or pred not in label2idx:
            continue
        else:
            agreement_table[label2idx[ref]][label2idx[pred]] += 1

    print('Ready to make confusion matrix')
    confusion_matrix = sns.heatmap(
        agreement_table,
        annot=True,
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        fmt='.0f',
        ax=ax,
        cbar=False
    )
    fig = confusion_matrix.get_figure()
    fig.savefig(outpath)


def make_confusion_matrix():
    parser = ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("predictions")
    parser.add_argument("outdir")
    args = parser.parse_args()

    # TODO: Make confusion matrix
    ref_labels = []
    pred_labels = []

    bcms_label_set = {'bs', 'hr', 'me', 'sr'}
    eng_label_set = {'EN-GB', 'EN-US'}
    spanish_label_set = {'ES-AR', 'ES-ES'}
    port_label_set = {'PT-PT', 'PT-BR'}
    french_label_set = {'FR-FR', 'FR-CH', 'FR-CA', 'FR-BE'}

    with open(args.reference, 'r', encoding='utf8') as ref:
        for line in ref:
            ref_labels.append(line.strip())
            line = line.strip()
            if line.startswith("EN"):
                eng_label_set.add(line)
            elif line.startswith("FR"):
                french_label_set.add(line)
            elif line.startswith("PT"):
                port_label_set.add(line)
            elif line.startswith("ES"):
                spanish_label_set.add(line)
            else:
                bcms_label_set.add(line)

    bcms_label_set = sorted(bcms_label_set)
    eng_label_set = sorted(eng_label_set)
    spanish_label_set = sorted(spanish_label_set)
    port_label_set = sorted(port_label_set)
    french_label_set = sorted(french_label_set)


    with open(args.predictions, 'r', encoding='utf8') as preds:
        for line in preds:
            if not line.strip():
                # There may be cases where no prediction passes a threshold
                line = None
                pred_labels.append(line)
            else:
                pred_labels.append(line.strip())

    print('Files read ok')
    assert len(ref_labels) == len(pred_labels), f"{len(ref_labels)} refs and {len(pred_labels)}"

    plt.rcParams["figure.figsize"] = [7.50, 7.50]
    plt.rcParams["figure.autolayout"] = True

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
    fig.subplots_adjust(wspace=0.01)

    os.makedirs(args.outdir, exist_ok=True)
    write_confusion_matrix(bcms_label_set, ref_labels, pred_labels, os.path.join(args.outdir, 'bcms-confusion.png'), ax1)
    write_confusion_matrix(french_label_set, ref_labels, pred_labels,
                           os.path.join(args.outdir, 'french-confusion.png'), ax2)
    write_confusion_matrix(eng_label_set, ref_labels, pred_labels,
                           os.path.join(args.outdir, 'english-confusion.png'), ax3)
    write_confusion_matrix(spanish_label_set, ref_labels, pred_labels,
                           os.path.join(args.outdir, 'spansish-confusion.png'), ax4)
    write_confusion_matrix(port_label_set, ref_labels, pred_labels,
                           os.path.join(args.outdir, 'portuguese-confusion.png'), ax5)

    ax6.axis('off')
    plt.xlabel("Predicted Label")
    plt.ylabel("Correct Label")
    fig.subplots_adjust(wspace=0.001)

    plt.savefig('all_matrices.png')

if __name__ == "__main__":
    make_confusion_matrix()