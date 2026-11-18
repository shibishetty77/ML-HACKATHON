# Four-feature MLP experiment

These files archive Team 9's completed MLP run on the same shared candidate
tables and trip folds as XGBoost. The model has 4 input features, two ReLU
layers of 32 and 16 units, and a binary sigmoid output. Fold-specific
`StandardScaler` instances are fitted only on training rows. The final model
was refitted on all 80 training trips and evaluated on 10 student-test trips.

`train_mlp.py` is the supplied `train_teammate_mlp.py` with only its input and
output paths changed to this repository layout. The saved metrics, predictions,
histories and model artifact were copied from the supplied completed run.
`score_existing.py` checks the archived student-test probabilities against the
shared candidate CSV and recomputes candidate F1, log loss and Top-1 values.

Run from the repository root with Python 3.11:

```powershell
py -3.11 analysis\mlp_shared\train_mlp.py
```

This full retraining writes new outputs to this directory and can take several
minutes. The original member's separate 16-feature MLP investigation used a
different candidate generator and is not mixed into the common comparison.
