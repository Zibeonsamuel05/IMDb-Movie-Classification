# Model Comparison Report

## Objective
The project compares classification algorithms for predicting categories derived from IMDb movie ratings.

## Classification tasks
**Multiclass:** Poor (<5), Average (5–<7), Good (7–<8.5), Excellent (8.5–10).

**Binary:** Low-rated (<7) and High-rated (≥7).

## Results supplied for the project
| Model | Task | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Multiclass | 0.550 | 0.615 | 0.550 | 0.574 |
| Decision Tree | Multiclass | 0.560 | 0.598 | 0.560 | 0.574 |
| Random Forest | Multiclass | 0.658 | 0.628 | 0.658 | 0.637 |
| KNN | Binary | 0.645 | 0.636 | 0.645 | 0.636 |
| SVM | Binary | 0.740 | 0.738 | 0.740 | 0.739 |

*Precision, recall and F1 are recorded as weighted averages in the reproducible pipeline. Confirm that this matches the original evaluation code before final submission.*

## Interpretation
Random Forest was the strongest of the three evaluated multiclass models. An ensemble of trees can represent nonlinear feature interactions and is generally less dependent on one tree's particular split decisions than a single Decision Tree.

SVM was the strongest of the two evaluated binary models. An RBF-kernel SVM can model nonlinear decision boundaries, while its margin-based objective and regularisation can support generalisation. These are plausible explanations, not proof of the exact cause of the observed scores.

## Conclusion
Based on the supplied results, Random Forest is the best-performing multiclass model (65.8% accuracy; 0.637 F1), while SVM is the best-performing binary model (74.0% accuracy; 0.739 F1). The two tasks use different target structures, so their scores should not be compared as if they were the same task.

## Reproducibility note
The table above records the metrics supplied during project preparation. Run the notebook on the exact Kaggle CSV to obtain fresh metrics. If the results differ, update this report and the chart using the newly generated outputs before creating a public release.
