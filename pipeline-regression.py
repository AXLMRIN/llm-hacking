import os
import json 

import pandas as pd 
from torch import load

from regression import perform_regression

def prepare_data(path_result: str, model_iteration: str):
    """
    Load labels and join metadata and create dummies for regression
    Return metadata from the label
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
    x_columns_list = [f"{label_dichotomized}-GS",f"{label_dichotomized}-PRED"]

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
    joined_df = pd.concat((joined_df, pd.DataFrame(dummy_columns)), axis = 1)

    # Load metadata from the label
    with open(f"{path_result}/{model_iteration}/scores.json", "r") as file:
        scores = json.load(file)
    best_epoch, best_score = sorted(scores.items(), key = lambda x : x[1], reverse = True)[0]
    training_args = load(f"{PATH_RESULT}/{model_iteration}/training_args.bin", weights_only=False)
    learning_rate = training_args.learning_rate
    weight_decay = training_args.weight_decay
    model_metadata = {
        "learning_rate": learning_rate, 
        "weight_decay": weight_decay, 
        "best_epoch": best_epoch, 
        "best_score": best_score, 
    }

    return joined_df, x_columns_list, model_metadata

PATH_RESULT = "./results/ideology_news_dichotomized"
MODEL_NAME = "answerdotai/ModernBERT-base"
full_results = {}
counter = 0
for model_iteration in os.listdir(PATH_RESULT):
    if not os.path.isdir(f"{PATH_RESULT}/{model_iteration}"): continue  # Skip for non directories
    
    df, x_columns_list, model_metadata = prepare_data(PATH_RESULT, model_iteration)
    # create_dummies:
    for topic in df["topic"].unique():
        df.loc[:,f"T-{topic}"] = (df["topic"] == topic).copy()
    for source in df["source"].unique():
        df.loc[:,f"S-{source}"] = (df["source"] == source).copy()

    y_columns_list = [col for col in df.columns if col.startswith("T-")] + [col for col in df.columns if col.startswith("S-")]
    for x_column in x_columns_list:
        for y_column in y_columns_list:
            full_results[counter] = {
                **perform_regression(df = df, y_column = y_column, x_column = x_column),
                "x_column" : x_column,
                "y_column" : y_column,
                "model" : MODEL_NAME,
                **model_metadata
            }
            counter += 1

with open(f"{PATH_RESULT}/full_results.json", "w") as file:
    json.dump(full_results, file, indent = 4, ensure_ascii = True)
