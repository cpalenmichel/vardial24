import os
from argparse import ArgumentParser

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, EvalPrediction

from multiclassbert import BertForMultilabelSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


def train_model():
    parser = ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--test", default=None)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--batchsize", default=8)
    parser.add_argument("--learning-rate", default=2e-5, type=float)
    parser.add_argument("--epochs", default=3, type=int)

    args = parser.parse_args()

    model_ckpt = args.model

    en_dataset_path_train = args.train
    en_dataset_path_dev = args.test if args.test else args.dev
    # pd_dataset = pd.read_csv(en_dataset_path, sep='\t')
    train_dataset = load_dataset('csv', data_files=[
        en_dataset_path_train,
    ], delimiter='\t',
                                 column_names=["label", "text"]
                                 )
    dev_dataset = load_dataset('csv', data_files=[
        en_dataset_path_dev,
    ], delimiter='\t',
                               column_names=["label", "text"]
                               )
    if not args.test:
        en_dataset = DatasetDict({"train": train_dataset["train"], "test": dev_dataset['train'],
                              "validation": dev_dataset['train']})
    else:
        test_dataset = load_dataset('csv', data_files=[
            args.test,
        ], delimiter='\t',
                                   column_names=["label", "text"]
                                   )
        en_dataset = DatasetDict({"train": train_dataset["train"], "test": test_dataset,
                                  "validation": dev_dataset['train']})

    label_set = set()
    for split in en_dataset:
        for example in en_dataset[split]:
            label_set.add(example["label"])

    id2label = {i: label for i, label in enumerate(sorted(label_set))}
    label2id = {label: idx for idx, label in id2label.items()}

    en_dataset = en_dataset.map(
        lambda x: {label: (1 if label in x["label"].split(',') else 0) for label in label2id}
    )

    en_dataset = en_dataset.map(
        lambda x: {"labels": [x[c] for c in label2id if c != "text" and c != "label"]})



    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize_and_encode(examples):
        return tokenizer(examples["text"], truncation=True)

    cols = en_dataset["train"].column_names
    cols.remove("labels")
    ds_enc = en_dataset.map(tokenize_and_encode, batched=True, remove_columns=cols)

    num_labels = len(label2id)
    model = BertForMultilabelSequenceClassification.from_pretrained(model_ckpt,
                                                                    num_labels=num_labels)

    # def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    #     # TODO: Need to switch this to compute Macro-F1 or get setup to just run inference and score later.
    #
    #     y_pred = torch.from_numpy(y_pred)
    #     y_true = torch.from_numpy(y_true)
    #     if sigmoid:
    #         y_pred = y_pred.sigmoid()
    #     return ((y_pred > thresh) == y_true.bool()).float().mean().item()
    #
    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     # TODO: Macro / micro f1 and prec recall
    #     return {'accuracy_thresh': accuracy_thresh(predictions, labels)}

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1-micro': f1_micro_average,
                    'f1-macro': f1_macro,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result


    batch_size = args.batchsize

    training_args = TrainingArguments(
        output_dir=args.outdir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=ds_enc["train"],
        eval_dataset=ds_enc["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer)

    trainer.train()
    trainer.evaluate()
    predictions = trainer.predict(test_dataset=ds_enc["test"])
    
    with open(os.path.join(args.outdir, 'predictions.txt'), 'w', encoding='utf8') as outfile:
        for pred in predictions.predictions:
            print(",".join([id2label[idx]  for idx in range(len(pred)) if pred[idx] == 1]), file=outfile)

if __name__ == "__main__":
    train_model()