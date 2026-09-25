# Notebooks

Exploratory notebooks are intentionally not committed: they go stale, they carry
output noise into diffs, and everything needed to reproduce a result lives in
`src/`. `.gitignore` excludes `notebooks/*.ipynb` for that reason.

## Suggested exploration order

1. **Data profile** - load `DATA_DIR/diabetes_risk_prediction_dataset.csv`, check
   the missing-value pattern (several numeric columns carry 2-6% missing values)
   and the class balance (the `Low` band is under 1% of rows).
2. **Leakage check** - cross-tabulate `Diabetes_Risk_Score` against
   `Diabetes_Risk`. The score maps onto the label by fixed cut-offs, which is why
   `src.config.LEAKAGE_COLUMNS` excludes it along with `AI_Health_Recommendation`
   and `Doctor_Consultation_Needed`.
3. **Baseline** - reuse the project code rather than re-implementing it:

   ```python
   from src.data import load_raw_dataset, clean_dataset, split_dataset, split_features_target
   from src.features import build_pipeline

   frame = clean_dataset(load_raw_dataset())
   train_df, test_df = split_dataset(frame)
   features, labels = split_features_target(train_df)
   pipeline, numeric, categorical = build_pipeline(features, "logreg")
   pipeline.fit(features, labels)
   ```

4. **Error analysis** - inspect the `Moderate` vs `High` boundary, which is where
   the confusion matrix from `python -m src.evaluate` concentrates its mistakes.

Start Jupyter from the project root so that `import src` resolves:

```bash
pip install jupyterlab
jupyter lab
```
