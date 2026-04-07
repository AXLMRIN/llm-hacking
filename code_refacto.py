"""
In : 
    configuration file + input csvs (NO PREPROCESSING OF THE TEXT NOR SAMPLING IN 
    THIS SCRIPT)
Out: 
    predictions as a CSV + csv of all configurations, scores and path to relevant 
    data 
"""
import os 
from time import time
import json
from itertools import product
import pandas as pd 
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from functions import *
from toolbox.CustomLogger import CustomLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

prepare_environment()
logger = CustomLogger("./custom_logs")

TEST_MODE = True
BATCH_SIZE = 4
TOTAL_BATCH_SIZE = 16

with open("./config.json") as file:
    config_json = json.load(file)

parameter_names, parameters_values = extract_hyperparameters(config_json)

for dataset_info in config_json["datasets"]:
    df = pd.read_csv(dataset_info["filepath-train"])
    df = sanitize_df(df, **dataset_info)
    labels = list(df["LABEL"].unique())
    
    df_prediction = pd.read_csv(dataset_info["filepath-predict"])
    df_prediction = sanitize_df(df_prediction, **dataset_info)

    for label in labels: 
        task_name = f"{dataset_info['name']}-{label}"
        dichotomized_df, label2id, id2label = dichotomize(df, label)
        dichotomized_df_prediction, _, _ = dichotomize(df_prediction, label)
        for local_config in product(*parameters_values):
            logger("START LOOP" + "#" * 91, skip_line="before")
            model, tokenizer, ds_loop, dsd_loop, ds_pred, predictions = (None, )*6
            
            loop_config = {n: v for n,v in zip(parameter_names,local_config)}
            logger(f"Starting Loop on task {task_name} {'(TEST_MODE)' if TEST_MODE else ''} and config {loop_config}")

            loop_ID = {
                **loop_config, 
                "task_name": task_name,
                "dataset_train": dataset_info["filepath-train"],
                "dataset_predict": dataset_info["filepath-predict"],
            }
            hash_ = create_hash(**loop_ID)
            if already_done(hash_):
                logger("Loop was already completed")
                logger("END LOOP" + "#" * 92)
                continue

            try: 
                # Prepare tokenizer: model_name, context_window_rel_to_max
                tokenizer = load_tokenizer(**loop_config)

                max_n_tokens = get_max_tokens(dichotomized_df["TEXT"], tokenizer)
                # ⚠️ How do we deal with entries longer than the model's context window
                max_length_capped = cap_max_length(max_n_tokens=max_n_tokens, **loop_config)
                tokenization_parameters = {
                    'padding' : 'max_length',
                    'truncation' : True,
                    'max_length' : max_length_capped
                }
                
                # Prepare dataset: N_train, train_eval_test_ratios
                ds_loop: Dataset = sample_N_elements(dichotomized_df, SEED = 0, **loop_config)
                dsd_loop : DatasetDict = split_ds(ds_loop, SEED = 0, **loop_config)
                dsd_loop = dsd_loop.map(lambda row: tokenize_dataset_dict(
                    row,
                    label2id, 
                    tokenizer,
                    tokenization_parameters
                ))

                # Prepare model: model_name
                model = AutoModelForSequenceClassification.from_pretrained(
                    loop_config["model_name"],
                    num_labels = len(label2id),
                    id2label   = id2label,
                    label2id   = label2id,
                )
                
                # Prepare trainer: learning_rate, weight_decay, warmup_ratio, dropout
                output_dir = "./models/current"
                training_args = load_training_arguments(
                    output_dir=output_dir, 
                    batch_size_device=BATCH_SIZE, 
                    total_batch_size=TOTAL_BATCH_SIZE, 
                    **loop_config
                )
                logger("Everything loaded — Start training")
                tstart = time()
                best_model_checkpoint = train_model(
                    model, 
                    training_args,
                    dsd_loop,
                    TEST_MODE
                )
                logger(f"Training done in {time() - tstart():.0f}s")

                model = None
                
                # Reload model from checkpoint
                model = AutoModelForSequenceClassification.from_pretrained(best_model_checkpoint)
                predictions : pd.DataFrame = predict(model, dsd_loop["test"], batch_size=BATCH_SIZE, id2label=id2label)
                score_on_test = f1_score(y_true = predictions["GS-LABEL"], y_pred = predictions["PRED-LABEL"], average="macro",zero_division=np.nan)
                logger(f"Evaluate best model. Score: {score_on_test}")

                # Predict on full data
                ds_pred = Dataset.from_pandas(df_prediction)
                
                if TEST_MODE : ds_pred = ds_pred.select(range(50))

                ds_pred = ds_pred.map(lambda row: tokenize_dataset_dict(
                    row,
                    label2id, 
                    tokenizer,
                    tokenization_parameters
                ))

                logger("Start Inference")
                tsart = time()
                predictions : pd.DataFrame = predict(model, ds_pred, batch_size=BATCH_SIZE, id2label=id2label)
                logger(f"Inference done in {time() - tstart:.0f} s")

                if not TEST_MODE:
                    predictions.to_csv(f"./predictions_save/{hash_}.csv")
                    to_save = {
                        **loop_ID,
                        "score_on_test": score_on_test,
                        "prediction-csv": f"./predictions_save/{hash_}.csv"
                    }                    

                    to_saving_logs(hash_, to_save)
                    logger(f"Information saved with hash {hash_}")
            
            except Exception as e:print(f"Error in loop\n{e}")
            finally: 
                del tokenizer, ds_loop, dsd_loop, ds_pred, predictions #TODO: re-delete model
                clean() 
                logger("END LOOP" + "#" * 92)

            break
        break

#TODO
"""
- remove the breaks
- Add seed to training ???
- CHECK FOR MISTAKES
- How to for large texts small window size
- implement pooling strategies
"""