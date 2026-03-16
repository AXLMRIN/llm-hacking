# Documentation for ideology_news dataset — Axel

**Downloaded from:** [Github/Article Bias Prediction](https://github.com/ramybaly/Article-Bias-Prediction/tree/main/) (2026-02-17)

**Description**

- Columns of interest: topic (108 unique, ex: "politics", "china", don't know how relevant and reliable); source (491 unique, ex: "Politico", "Bloomberg", don't know how relevant and reliable); "bias" ("left", "right", "center"); content (text clean); ID; date (publication date, don't know how reliable, but mostly clean).
- classification task: classify the content as "left", "right" or "center" (dichotomized)
- texts length: 
```python 
df_raw["content"].apply(len).describe()
# count    37554.000000
# mean      5847.792006
# std       4001.537702
# min        131.000000
# 25%       3355.000000
# 50%       4949.000000
# 75%       7020.000000
# max      53992.000000
# Name: content, dtype: float64
```

**Preprocessing**

Preprocess file: `preprocessing-data/preprocess-ideology_news.ipynb`:

- text content is clean
- remove inputs where the date is missing or invalid
- only keep content of roughly the same length

```python
condition = np.logical_and.reduce((
    df_raw["year"].notna(),
    df_raw["month"].notna(),
    df_raw["year"] != 1,
    df_raw["year"] != 2050,
    df_raw["year"] != 2001, # Not enough elements  
    df_raw["year"] != 2007, # Not enough elements
    df_raw["year"] != 2010,  # Not enough elements
    df_raw["content"].apply(len) <= 7000, # q75
    df_raw["content"].apply(len) >= 3300, # q25
))

df = df_raw[condition]
```

After manipulation, tehre are about 1500 articles per year (2012-2020):
```python
df["year"].value_counts().sort_index()
# year
# 2012    1321
# 2013    2381
# 2014    1812
# 2015    1490
# 2016    1819
# 2017    1805
# 2018    2024
# 2019    2106
# 2020    1791
# Name: count, dtype: int64
```

- add split metadata using the splits available on the repo (not really used afterwards but they're here)
- Create a stratified (by year and bias) dataset for training (N=4050).<br/>File: `ideology_news-stratified_year_balanced.csv` <br/>**NOTE: this might be unwise because we train on a dichotomized scheme, so creating a balanced trainset (1/3, 1/3, 1/3) will result in a 2/3 - 1/3 balance after dichotomization.**.
- Create a dataset for inference (N = 10,000) (train set and inference set are mutually exclusive)<br/>File: `ideology_news-dataset_for_inference.csv`

After creating the datasets I have dichotomized each label, for both datasets.

For the inference dataset, I have created a specific folder to match the Training pipeline (`make-dataset-dict-for-inference.py`). This is basically creating a `DatasetDict` with a train, valid and test split and shoving all items in the `test-split`. Then , I encode the 10,000 text inputs with the appropriate tokenizer.

**Training**

File: `pipeline-ideology_news.py`

For each bias, I train a bert model on 4 epochs (split train and validation) and evaluate the F1 score (macro). For the best performing model across epochs, I predict the labels on the inference set. For all epochs, I delete the model weights as they are quite heavy. This generates CSV files `test_labels-XXXX.csv` with the following columns: `"ID", "LABEL-GS", "LABEL-PRED"`.

I repeath the pipeline for the following hyperparameters: 

- model: `["answerdotai/ModernBERT-base"]`
- learning_rate: `[5e-4, 1e-4, 1e-5, 5e-5]`
- weight_decay: `[0.1, 0.05, 0.01]`

total number of pipelines: 36 (= $\underbrace{3}_{bias} \times \underbrace{1}_{model} \times \underbrace{4}_{learning\ rate} \times \underbrace{3}_{weight\ decay}$`)

**Regression**

File: `pipeline-regression.py`

For each model (N = 36), I perform 364 logistic regressions (246 sources[^1], 108 topics) twice, once for the predicted labels ($label_{pred}\sim source|topic$) and once for the gold standard labels ($label_{pred}\sim source|topic$) which results in 25,488 regressions ($= \underbrace{36}_{fine\ tuned\ models}\times \underbrace{2}_{predicted | gold standard}\times(\underbrace{246}_{sources} + \underbrace{108}_{topics})$). 

Each column for the regression is binarised as integers (0 or 1).

For each regression I saved the following: 

- x_column
- y_column
- Pseudo R-squared
- Covariate Names
- Coef
- Std err
- z
- pvalues
- Conf Int
- Log-Likelihood
- LL-Null
- LLR p-value
- AIC
- BIC
- N obs
- N iterations

as well as metadata from the model: 

- model _(name)_
- score on sample (f1 macro score on the 10,000 samples)
- learning_rate
- weight_decay
- best_epoch
- best_score (f1 macro score on the test set after training)

[^1]: This figure differs from the one presented in the **Description** (491), this is a result of many sources with only one article in the whole corpus. The 246 sources correspond to the sources of the 10,000 articles from the inference set.

**First results analysis**

File: `analyse-results-ideology-news.ipynb`

I first filter out unsuccessful and insignificant regressions:

- Out of 25,488 regressions, only 23,571 were successful (91% of all regressions)
- Out of 23,571 regressions, only 6640 were significant — i.e. `LLR p-value < 0.05` (28% of successful regressions, 25% of all regressions)
- Out of 6640 successful and significant regressions, only 2219 regressions have their counterpart (PRED - GS) (9% of all regressions)

For the 2219 cases where the regression is significant for the GS labels and predicted labels, I evaluate the errors of type 1, 2 and S. 

I get the following results: 

|Error type|Amount|Percentage (out of 2219)|
|---|---|---|
|Type 1|1129|51%|
|Type 2|9|<1%|
|Type S|2|<1%|