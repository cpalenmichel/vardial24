"""
Get some descriptive stats on the DSL VarDial dataset. Tokens, sentences, etc.
Uses spaCy to do so.
"""
from argparse import ArgumentParser

import spacy
from numpy import average
from numpy.core import std


def dataset_stats():
    parser = ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument("--language", choices = ["EN", "ES", "PT", "FR", "BCMS"], required=True)
    args = parser.parse_args()

    if args.language == "EN":
        nlp = spacy.load("en_core_web_sm")
    elif args.language == "ES":
        nlp = spacy.load("es_core_news_sm")
    elif args.language == "PT":
        nlp = spacy.load("pt_core_news_sm")
    elif args.language == "BCMS":
        nlp = spacy.load("hr_core_news_sm")
    elif args.language == "FR":
        nlp = spacy.load("fr_core_news_sm")
    else:
        raise ValueError("A language must be specified for parsing")

    with open(args.datafile, 'r', encoding='utf8') as infile:
        num_docs = 0
        sents_per_doc = []
        tokens_per_doc = []
        for line in infile:
            doc = line.strip()
            if not doc:
                continue
            parts = doc.split("\t")
            label = parts[0]
            text = parts[-1]
            num_docs += 1
            spacy_doc = nlp(text)
            sent_count = len([sent for sent in spacy_doc.sents])
            sents_per_doc.append(sent_count)
            token_count = len([token for sent in spacy_doc.sents for token in sent])
            tokens_per_doc.append(token_count)
        print(f"Language\tnum_docs\tmean_token_per_doc\ttoken_per_doc_std\tsents_per_doc")
        print(f"{args.language}\t{num_docs}\t{average(tokens_per_doc)}\t{std(tokens_per_doc)}\t{average(sents_per_doc)}\t{std(sents_per_doc)}")


if __name__ == "__main__":
    dataset_stats()