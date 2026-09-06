## 2D Branch

Early stopping: val AUC flat for 10 epochs.
[seed 42] test AUC 0.9793

** model parameters: 115,459 | train samples: 50968 (2.3 params/sample)

--- Floors (test set) ---

majority-class (0) accuracy 0.5000 balanced 0.5000 AUC 0.5000 n=9548

per-seed test AUC: ['0.9793']
mean 0.9793 std 0.0000 spread 0.0000

--- Event/noise detector [channels=2d, fusion=linear] (1-seed ensemble, test set) ---
n 9548
accuracy 0.9328
balanced_accuracy 0.9328
precision 0.9561
recall 0.9072
f1 0.9310
roc_auc 0.9793
pr_auc 0.9828
mcc 0.8667
brier 0.0556
log_loss 0.2231
confusion_matrix:
[4575, 199]
[443, 4331]
TN=4575, FP=199, FN=443, TP=4331

ROC-AUC 0.9793 vs majority-class floor 0.5000 -> BEATS floor

Classification Report (ensemble):

| Classification | precision | recall | f1-score | support |
| -------------- | --------- | ------ | -------- | ------- |
| 0.0            | 0.9117    | 0.9583 | 0.9344   | 4774    |
| 1.0            | 0.9561    | 0.9072 | 0.9310   | 4774    |

macro avg 0.9339 0.9328 0.9327 9548
weighted avg 0.9339 0.9328 0.9327 9548

## Gate fusion

Early stopping: val AUC flat for 10 epochs.
[seed 42] test AUC 0.9752

gate g (1 favors 1D, 0 favors 2D): mean 0.155 std 0.169
earthquake windows: mean g 0.090
noise windows: mean g 0.220
correct predictions: mean g 0.153
wrong predictions: mean g 0.172

model parameters: 191,874 | train samples: 50968 (3.8 params/sample)

--- Floors (test set) ---
majority-class (0) accuracy 0.5000 balanced 0.5000 AUC 0.5000 n=9548

per-seed test AUC: ['0.9752']
mean 0.9752 std 0.0000 spread 0.0000

--- Event/noise detector [channels=all, fusion=gate] (1-seed ensemble, test set) ---
n 9548
accuracy 0.9283
balanced_accuracy 0.9283
precision 0.9398
recall 0.9152
f1 0.9273
roc_auc 0.9752
pr_auc 0.9796
mcc 0.8568
brier 0.0590
log_loss 0.2326
confusion_matrix:
[4494, 280]
[405, 4369]
TN=4494, FP=280, FN=405, TP=4369

ROC-AUC 0.9752 vs majority-class floor 0.5000 -> BEATS floor

Classification Report (ensemble):
precision recall f1-score support

         0.0     0.9173    0.9413    0.9292      4774
         1.0     0.9398    0.9152    0.9273      4774

    accuracy                         0.9283      9548

macro avg 0.9286 0.9283 0.9282 9548
weighted avg 0.9286 0.9283 0.9282 9548
