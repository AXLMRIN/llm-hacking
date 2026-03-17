import os
import json 

import pandas as pd 
from sklearn.metrics import f1_score
import statsmodels.api as sm
from torch import load

def perform_regression(df: pd.DataFrame, y_column: str, x_column: str
) -> dict[str:str|float|list[float]]:
    """
    Perform logit regression and returns key values
    """
    Y = df[y_column].to_numpy().astype(int) # [1, 0, 0, 1, ...]
    X = df[x_column].to_numpy().astype(int) # [1, 0, 1, 0, ...]
    X = sm.add_constant(X)
    try: 
        model = sm.Logit(Y,X,)
        res = model.fit(maxiter=100)

        return {
            "Pseudo R-squared": res.prsquared,
            "Covariate Names": model.exog_names,
            "Coef": res.params.tolist(),
            "Std err": res.bse.tolist(),
            "z": res.tvalues.tolist(),          # z-statistics
            "pvalues": res.pvalues.tolist(),
            "Conf Int": res.conf_int().tolist(),
            "Log-Likelihood": res.llf,
            "LL-Null": res.llnull,
            "LLR p-value": res.llr_pvalue,
            "AIC": res.aic,
            "BIC": res.bic,
            "N obs": res.nobs,
            "N iterations": res.mle_retvals["iterations"],
        }
    except: 
        return {
            "Pseudo R-squared": "FAILED",
            "Covariate Names": "FAILED", 
            "Coef": "FAILED",
            "Std err": "FAILED",
            "z": "FAILED",          # z-statistics
            "pvalues": "FAILED",
            "Conf Int": "FAILED",
            "Log-Likelihood": "FAILED",
            "LL-Null": "FAILED",
            "LLR p-value": "FAILED",
            "AIC": "FAILED",
            "BIC": "FAILED",
            "N obs": "FAILED",
            "N iterations": "FAILED",
        } 
    
def prepare_data(path_result: str, model_iteration: str
) -> tuple[pd.DataFrame, list[str], list[str], dict[str:float|int|str]]:
    """
    Load labels and join metadata and create dummies for regression

    The folder {path_result}/{model_iteration}/ must contain the following: 
        - test_labels-0000.csv : with the ID column, a LABEL-GS and LABEL-PRED column
        - ...
        - test_labels-XXXX.csv : with the ID column, a LABEL-GS and LABEL-PRED column
        - DataHandler_config.json: with at least the "label_column"
        - scores.json: a dictionary binding the epoch to the F1 score of the model on the testset
        - training_args.bin: the file with the training arguments of the model

    Return dataframe with the joined dummies, the name of the dependant variables, 
    the name for the independant variables (dummies) and metadata from the label
    """
    # Load df labels
    df_labels = pd.concat([pd.read_csv(f"{path_result}/{model_iteration}/{filename}")
        for filename in os.listdir(f"{path_result}/{model_iteration}/")
        if filename.endswith(".csv") and filename.startswith("test_labels-")
    ])

    with open(f"{path_result}/{model_iteration}/DataHandler_config.json", "r") as file:
        label_dichotomized = json.load(file)["label_column"]
    df_labels = df_labels.rename(columns={
        "LABEL-GS" : f"{label_dichotomized}-GS",
        "LABEL-PRED" : f"{label_dichotomized}-PRED",
    })
    dependant_variables_columns = [f"{label_dichotomized}-GS",f"{label_dichotomized}-PRED"]

    # Load metadata df
    df_metadata = pd.read_csv("../Article-Bias-Prediction/data_agg.csv")
    # Join dfs
    joined_df = (
        df_labels
        .set_index("ID")
        .join(df_metadata.set_index("ID")[["topic", "source"]])
    )
    # Create dummies: 
    dummy_columns = {}
    for topic in joined_df["topic"].unique():
        dummy_columns[f"T-{topic}"] = (joined_df["topic"] == topic).copy()
    for source in joined_df["source"].unique():
        dummy_columns[f"S-{source}"] = (joined_df["source"] == source).copy()
    independant_variables_columns = list(dummy_columns.keys())
    joined_df = pd.concat((joined_df, pd.DataFrame(dummy_columns)), axis = 1)

    # Load metadata from the label
    with open(f"{path_result}/{model_iteration}/scores.json", "r") as file:
        scores = json.load(file)
    best_epoch, best_score = sorted(scores.items(), key = lambda x : x[1], reverse = True)[0]
    training_args = load(f"{path_result}/{model_iteration}/training_args.bin", weights_only=False)
    learning_rate = training_args.learning_rate
    weight_decay = training_args.weight_decay

    # Evaluate F1 score on samble: 
    score_on_current_sample = f1_score(
        y_true = joined_df[f"{label_dichotomized}-GS"].to_numpy(),
        y_pred = joined_df[f"{label_dichotomized}-PRED"].to_numpy(),
        average="macro"
    )

    model_metadata = {
        "label_dichotomized": label_dichotomized, 
        "learning_rate": learning_rate, 
        "weight_decay": weight_decay, 
        "best_epoch": best_epoch, 
        "best_score": best_score, 
        "score_on_sample": score_on_current_sample,
    }

    return (
        joined_df, 
        dependant_variables_columns,
        independant_variables_columns,
        model_metadata
    )