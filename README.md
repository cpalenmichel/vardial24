# vardial24
VarDial 2024 experiments



## Baseline Runs

hyperparams: 
`python train.py --train "/home/cpm/repos/DSL-ML-2024/EN/EN_train.tsv" --dev "/home/cpm/repos/DSL-ML-2024/EN/EN_dev.tsv" --outdir "/home/cpm/expts/vardial24/mbert/eng/" --model "google-bert/bert-base-multilingual-cased" --epochs 3 --batchsize 16`
lr: 2e-5

Dev runs: 
### English:

#### Scores for entire dataset (599 instances):
F1-score for class EN-GB: 82.69%
F1-score for class EN-US: 85.68%
Macro-avg F1 score:    84.18%
Weighted-avg F1 score: 84.41%

#### Scores for ambiguous subset (76 instances):

F1-score for class EN-GB: 78.40%
F1-score for class EN-US: 77.42%
Macro-avg F1 score:    77.91%
Weighted-avg F1 score: 77.91%

### Spanish

Scores for entire dataset (989 instances):

F1-score for class ES-AR: 78.36%
F1-score for class ES-ES: 86.35%
Macro-avg F1 score:    82.36%
Weighted-avg F1 score: 83.02%

Scores for ambiguous subset (318 instances):

F1-score for class ES-AR: 84.78%
F1-score for class ES-ES: 87.23%
Macro-avg F1 score:    86.01%
Weighted-avg F1 score: 86.01%

### Portuguese

Scores for entire dataset (991 instances):
F1-score for class PT-BR: 80.58%
F1-score for class PT-PT: 68.32%
Macro-avg F1 score:    74.45%
Weighted-avg F1 score: 76.19%

Scores for ambiguous subset (134 instances):
F1-score for class PT-BR: 69.90%
F1-score for class PT-PT: 74.77%
Macro-avg F1 score:    72.33%
Weighted-avg F1 score: 72.33%

### French
Scores for entire dataset (17048 instances):
F1-score for class FR-BE: 98.63%
F1-score for class FR-CA: 96.08%
F1-score for class FR-CH: 94.27%
F1-score for class FR-FR: 97.36%
Macro-avg F1 score:    96.58%
Weighted-avg F1 score: 97.56%

Scores for ambiguous subset (120 instances):
F1-score for class FR-BE: 88.89%
F1-score for class FR-CA: 40.00%
F1-score for class FR-CH: 0.00%
F1-score for class FR-FR: 44.64%
Macro-avg F1 score:    43.38%
Weighted-avg F1 score: 59.44%

### Slavics
Scores for entire dataset (122 instances):
F1-score for class bs: 0.00%
F1-score for class hr: 0.00%
F1-score for class me: 0.00%
F1-score for class sr: 82.69%
Macro-avg F1 score:    20.67%
Weighted-avg F1 score: 47.73%

Scores for ambiguous subset (25 instances):
F1-score for class bs: 0.00%
F1-score for class hr: 0.00%
F1-score for class me: 0.00%
F1-score for class sr: 78.05%
Macro-avg F1 score:    19.51%
Weighted-avg F1 score: 24.02%


### Concatenated (seems worse...?)


### Data Augmentation: 
ENG NAIVE
Evaluated file: /home/cpm/expts/vardial24/mbert/eng-naive/predictions.txt

Scores for entire dataset (599 instances):
F1-score for class EN-GB: 80.27%
F1-score for class EN-US: 84.68%
Macro-avg F1 score:    82.47%
Weighted-avg F1 score: 82.80%

Scores for ambiguous subset (76 instances):
F1-score for class EN-GB: 76.42%
F1-score for class EN-US: 81.25%
Macro-avg F1 score:    78.84%
Weighted-avg F1 score: 78.84%

ENG TREE MIX
Evaluated file: /home/cpm/expts/vardial24/mbert/eng-tree/predictions.txt

Scores for entire dataset (599 instances):
F1-score for class EN-GB: 78.49%
F1-score for class EN-US: 85.12%
Macro-avg F1 score:    81.80%
Weighted-avg F1 score: 82.30%

Scores for ambiguous subset (76 instances):
F1-score for class EN-GB: 73.33%
F1-score for class EN-US: 81.25%
Macro-avg F1 score:    77.29%
Weighted-avg F1 score: 77.29%


ES NAIVE AUG
Scores for entire dataset (989 instances):
F1-score for class ES-AR: 77.15%
F1-score for class ES-ES: 87.03%
Macro-avg F1 score:    82.09%
Weighted-avg F1 score: 82.91%

Scores for ambiguous subset (318 instances):
F1-score for class ES-AR: 83.52%
F1-score for class ES-ES: 92.75%
Macro-avg F1 score:    88.13%
Weighted-avg F1 score: 88.13%

ES TREE MIX 
Evaluated file: /home/cpm/expts/vardial24/mbert/spa-tree/predictions.txt

Scores for entire dataset (989 instances):
F1-score for class ES-AR: 75.53%
F1-score for class ES-ES: 86.85%
Macro-avg F1 score:    81.19%
Weighted-avg F1 score: 82.13%

Scores for ambiguous subset (318 instances):
F1-score for class ES-AR: 81.78%
F1-score for class ES-ES: 91.10%
Macro-avg F1 score:    86.44%
Weighted-avg F1 score: 86.44%

PT NAIVE
Evaluated file: /home/cpm/expts/vardial24/mbert/por-naive/predictions.txt

Scores for entire dataset (991 instances):
F1-score for class PT-BR: 83.05%
F1-score for class PT-PT: 69.04%
Macro-avg F1 score:    76.05%
Weighted-avg F1 score: 78.03%

Scores for ambiguous subset (134 instances):
F1-score for class PT-BR: 88.33%
F1-score for class PT-PT: 74.18%
Macro-avg F1 score:    81.26%
Weighted-avg F1 score: 81.26%

PT TREE 
Scores for entire dataset (991 instances):
F1-score for class PT-BR: 83.16%
F1-score for class PT-PT: 64.23%
Macro-avg F1 score:    73.69%
Weighted-avg F1 score: 76.38%

Scores for ambiguous subset (134 instances):
F1-score for class PT-BR: 84.98%
F1-score for class PT-PT: 70.53%
Macro-avg F1 score:    77.75%
Weighted-avg F1 score: 77.75%
