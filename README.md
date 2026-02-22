# Portuguese Bank Marketing Analysis (Predictive Targeting)

Machine learning analysis of a Portuguese bank’s telemarketing campaign data to predict whether a client will subscribe to a term deposit (`y ∈ {yes,no}`), and to compare model performance across several classical ML approaches. 

---

## Project goals

- Replicate and extend the approach from Moro et al. (2014) on bank telemarketing success prediction.
- Benchmark multiple classification models on the same train/test split.
- Report performance using **AUC** (primary) plus supporting metrics (RMSE / error rates where applicable).
- Avoid **target leakage** by excluding `duration` from predictive models (call length is only known after contact). 

---

## Data

- **Dataset:** Portuguese banking institution telemarketing campaigns  
- **Size:** 41,188 observations × 21 variables  
- **Target:** `y` (term deposit subscription: yes/no)  
- **Notes:** No missing values reported in the working dataset.

---

## Methods

- **Split:** 80/20 train-test split (`set.seed(42)`)
- **Cross-validation:** 10-fold CV used where applicable (e.g., logistic baseline CV; glmnet CV; SVM tuning via CV)
- **Models compared:**
  - Logistic Regression (baseline “paper-like” + extended)
  - Ridge / LASSO (glmnet)
  - Decision Tree (baseline + extended)
  - Random Forest
  - LDA
  - KNN
  - SVM (linear + tuned; final reported as tuned radial)
  - Neural Network (single hidden layer) 

---

## Key results (test AUC)

Best overall model in this implementation:

- **Random Forest:** **AUC ≈ 0.802** (best among evaluated models) 

Other notable AUCs (approx.):

- LASSO: ~0.798  
- Ridge: ~0.796  
- LDA: ~0.796  
- Logistic (baseline/extended): ~0.789  
- Neural Network: ~0.790  
- Tuned SVM (radial): ~0.715  
- KNN: ~0.618 

**Takeaway:** Ensemble tree methods performed strongest here; overall AUCs were lower than the paper’s reported peak results, likely due to predictor differences and validation approach differences (k-fold CV vs rolling window in the original work).

---

**Outputs**

Final report (HTML): Final_Report.html 
