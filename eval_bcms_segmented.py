import os
from argparse import ArgumentParser

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from multiclassbert import BertForMultilabelSequenceClassification


def run_eval():

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--traindata")
    parser.add_argument("--evaldata")
    parser.add_argument("outdir")

    args = parser.parse_args()

    col_names = ["label", "text", "seg_idx"]
    model_ckpt = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    train_dataset = load_dataset('csv', data_files=[
        args.traindata,
    ], delimiter='\t',
                                 column_names=col_names
                                 )

    test_dataset = load_dataset('csv', data_files=[
        args.infilepath,
    ], delimiter='\t', column_names=col_names)

    def tokenize_and_encode(examples):
        return tokenizer(examples["text"], truncation=True)

    label_set = set()
    for example in train_dataset:
        for lab in example["label"].split(","):
            label_set.add(lab)

    id2label = {i: label for i, label in enumerate(sorted(label_set))}
    label2id = {label: idx for idx, label in id2label.items()}

    test_dataset = test_dataset.map(
        lambda x: {label: (1 if label in x["label"].split(',') else 0) for label in label2id}
    )

    test_dataset = test_dataset.map(
        lambda x: {"labels": [x[c] for c in label2id if c != "text" and c != "label"]})

    test_dataset = test_dataset.map(tokenize_and_encode, batched=True)

    num_labels = len(label2id)
    model = BertForMultilabelSequenceClassification.from_pretrained(model_ckpt,
                                                                    num_labels=num_labels)

    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        with open(os.path.join(args.outdir, 'bcms_segmented_predictions.txt'), 'w', encoding='utf8') as outfile:
            for ex in test_dataset:
                logits = model(ex.input_ids)
                probs = sigmoid(torch.Tensor(logits))
                threshed_pred = np.zeros(probs.shape)
                threshed_pred[np.where(probs >= 0.5)] = 1
                pred_labels = ",".join([id2label[idx] for idx in range(len(probs)) if threshed_pred[idx] == 1])
                print(f"{pred_labels}\t{ex['labels']}\t{ex['seg_idx']}", file=outfile)


if __name__ == "__main__":
    run_eval()