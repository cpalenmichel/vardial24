"""
Parse a sentence with Universal Dependency.
Swap subtrees to generate new examples, swap phrases from other labels.
"""
import os
from argparse import ArgumentParser
from collections import defaultdict
import random
from pathlib import Path

import spacy
from spacy.lang.en.examples import sentences
from spacy.tokens.token import Token

random.seed(42)


def syntactic_augment():
    parser = ArgumentParser()
    parser.add_argument("inpath")
    parser.add_argument("outpath")
    parser.add_argument(
        "--language"
    )
    parser.add_argument("--augment-ratio", default=1.0, type=float)
    args = parser.parse_args()


    if args.language == "EN":
        nlp = spacy.load("en_core_web_sm")
    elif args.language == "ES":
        nlp = spacy.load("es_core_news_sm")
    elif args.language == "PT":
        nlp = spacy.load("pt_core_news_sm")
    elif args.language == "BCMS":
        nlp = spacy.load("hr_core_news_sm")
    else:
        raise ValueError("A language must be specified for parsing")

    texts_by_label = defaultdict(list)
    labels_to_subtrees = defaultdict(lambda: defaultdict(list))
    with open(args.inpath, "r", encoding="utf8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            label = fields[0]
            text = fields[1]
            texts_by_label[label].append(text)

            doc = nlp(text)
            for sent in doc.sents:
                for token in sent:

                    subtree_tokens = [t for t in token.subtree]
                    if len(subtree_tokens) < len(sent) and len(subtree_tokens) > 2 and token.dep_ != "ROOT":
                        labels_to_subtrees[label][(str(token.dep_), token.pos_)].append((subtree_tokens, doc, sent))
    path = Path(args.outpath)
    os.makedirs(path.parent.absolute())
    with open(args.outpath, "w", encoding="utf8") as outfile:
        for label in texts_by_label:
            for text in texts_by_label[label]:
                old_text = nlp(text)
                for n in range(int(args.augment_ratio)):
                    for old_sent in old_text.sents:
                        # Sometimes the sentence spliter splits the existing sentences
                        if len(old_sent) < 4:
                            continue
                        rand_token: Token = random.choice(old_sent)
                        if not labels_to_subtrees[label][(rand_token.dep_, rand_token.pos_)]:
                            continue


                        subtree, new_doc, new_sent = random.choice(labels_to_subtrees[label][(str(rand_token.dep_), rand_token.pos_)])
                        tries = 0
                        if new_sent.text == old_sent.text and tries < 5:
                            subtree, new_doc, new_sent = random.choice(
                                labels_to_subtrees[label][
                                    (str(rand_token.dep_), rand_token.pos_)])
                            tries += 1

                        subtree_idxs = [t.idx for t in subtree]

                        span = new_doc.char_span(subtree_idxs[0], subtree_idxs[-1] + len(
                            subtree[-1].text))
                        replacement = span.text

                        old_subtree_toks = [t for t in rand_token.subtree]

                        chunk1 = old_text.text[old_sent.start_char: old_subtree_toks[0].idx]

                        second_chunk_idx = old_subtree_toks[-1].idx + len(old_subtree_toks[-1])

                        chunk2 = old_text.text[second_chunk_idx: old_sent.end_char]

                        aug_text = f"{chunk1}{replacement}{chunk2}"

                        print(f"{label}\t{aug_text}", file=outfile)



            # for token in doc:
            #     print(token.text, token.pos_, token.dep_)




if __name__ == "__main__":
    syntactic_augment()