import os
import json 

from regression import perform_regression, prepare_data

PATH_RESULT = "./results"
DATASET_NAME = "ideology_news_dichotomized"
MODEL_NAME = "ibm-granite/granite-embedding-small-english-r2"
PATH_TO_DATA = f"{PATH_RESULT}/{DATASET_NAME}/{MODEL_NAME}"
full_results = {}
counter = 0

for model_iteration in os.listdir(PATH_TO_DATA):

    if not os.path.isdir(f"{PATH_TO_DATA}/{model_iteration}"): 
        continue  # Skip for non directories
    
    (
        df, 
        dependant_variables_columns,
        independant_variables_columns, 
        model_metadata
    ) = prepare_data(PATH_TO_DATA, model_iteration)

    for dependant_variable in dependant_variables_columns:
        for independant_variable in independant_variables_columns:
            full_results[counter] = {
                "R-ID" : f"Reg-{counter:06}",
                "task" : f"{DATASET_NAME}-{model_metadata['label_dichotomized']}",
                "hypothesis": f"{model_metadata['label_dichotomized']}~{independant_variable}",
                "label-type": dependant_variable.split("-")[-1],
                "model" : MODEL_NAME,
                **model_metadata,
                "x_column" : independant_variable,
                "y_column" : dependant_variable,
                **perform_regression(
                    df = df, 
                    x_column = independant_variable,
                    y_column = dependant_variable, 
                ),
            }
            counter += 1

with open(f"{PATH_TO_DATA}-regression-results.json", "w") as file:
    json.dump(full_results, file, indent = 4, ensure_ascii = True)
