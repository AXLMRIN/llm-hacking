import os 
from pathlib import Path
from gc import collect as gc_collect
import hashlib 
import json

from datasets import Dataset, DatasetDict
import numpy as np 
import pandas as pd 
from torch import Tensor, device
from torch.cuda import is_available as cuda_available
from torch.cuda import empty_cache, synchronize, ipc_collect
from torch.backends.mps import is_available as mps_available

from transformers import (
    AutoTokenizer, 
    AutoConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.metrics import f1_score

def extract_hyperparameters(config_json: dict):
    """
    extract the names and values of hyperparameters
    """
    parameter_names = [
        *[name for name in config_json["data-hyperparameters"].keys()],
        *[name for name in config_json["model-hyperparameters"].keys()],
    ]
    parameters_values = [
        *[values for values in config_json["data-hyperparameters"].values()],
        *[values for values in config_json["model-hyperparameters"].values()],
    ]
    return parameter_names, parameters_values

def sanitize_df(df: pd.DataFrame, text_col: str, label_col:str, id_col:str, **kwargs)->pd.DataFrame:
    if not np.isin([text_col, label_col, id_col], df.columns).all():
        raise ValueError(
            f"The columns you provided cannot be found in the dataframe. "
            f"You provided: {[text_col, label_col, id_col]}. "
            f"The dataframe contains: {df.columns}"
        )
    df = df.rename(columns={
        text_col: "TEXT",
        label_col: "LABEL",
        id_col: "ID",
    })
    if np.array([
        df["ID"].isna().sum() > 0,
        df["TEXT"].isna().sum() > 0,
        df["LABEL"].isna().sum() > 0,
    ]).any():
        raise ValueError(
            f"Missing values: "
            f"\t ID: {df['ID'].isna().sum()}"
            f"\t TEXT: {df['TEXT'].isna().sum()}"
            f"\t LABEL: {df['LABEL'].isna().sum()}"
        )
    if df["ID"].is_unique:
        return df[["ID", "TEXT", "LABEL"]].set_index("ID")
    else:
        raise ValueError("ID column contains non-unique values.")

def dichotomize(df: pd.DataFrame, label:str) -> tuple[pd.DataFrame, dict[str:int], dict[int:str]]:
    if label not in df["LABEL"].values:
        raise ValueError(f"Label ({label}) not in df[\"LABEL\"]. "
                         f"Available labels: {df['LABEL'].unique()}")
    df["LABEL"] = (df["LABEL"] == label).replace({True:label, False:f"not-{label}"})
    label2id = {label:1, f"not-{label}": 0}
    id2label = {1:label, 0: f"not-{label}"}
    return df, label2id, id2label

def load_tokenizer(model_name: str, **kwargs):
    try: 
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    except Exception as e:
        raise ValueError("Could not load the Tokenizer.\nErreur:{e}")
    
def tokenize_string(text: str, tokenizer, tokenization_parameters : dict = {}):
    return tokenizer(text, **tokenization_parameters)
    
def get_max_tokens(texts: pd.Series, tokenizer, top_n : int = 15)->int:
    """
    Tokenize the top_n longest entries (in term of characters), tokenize them and 
    return the maximum length of the encoded sentences
    """
    longests_as_index = texts.apply(len).sort_values(ascending=False).head(top_n).index
    max_tokenizing_len = (
        texts[longests_as_index]
        .apply(lambda txt: tokenize_string(txt, tokenizer)["input_ids"])
        .apply(len)
        .max()
    )
    return max_tokenizing_len

def pick_seed(**kwargs)->int:
    if "SEED" in kwargs:
        return kwargs["SEED"]
    else:
        return 42

def cap_max_length(max_n_tokens : int, context_window_rel_to_max : int, model_name : str, **kwargs):
    requested = context_window_rel_to_max * max_n_tokens / 100
    model_max = AutoConfig.from_pretrained(model_name).max_position_embeddings
    return min(requested, model_max)

def sample_N_elements(df: pd.DataFrame, N_train: int, **kwargs)->pd.DataFrame:
    """
    Sample N_elements
    """
    return Dataset.from_pandas(df.sample(N_train, random_state=pick_seed(**kwargs)))

def split_ds(ds : Dataset, train_eval_test_ratios : list[int], **kwargs)-> DatasetDict:
    """
    takes the train_eval_test_ratios (ex: [80, 10, 10]) and return a DatasetDict
    """
    if len(train_eval_test_ratios) != 3:
        raise ValueError(
            f"There should be three ints in train_eval_test_ratios. Found: " 
            f"{train_eval_test_ratios}"
        )
    if sum(train_eval_test_ratios) != 100:
        raise ValueError(
            f"The sum of train_eval_test_ratios shoul be 100. Found: "
            f"{train_eval_test_ratios}"
        )
    out_dsd = ds.train_test_split(
        train_size= train_eval_test_ratios[0] / 100, # Train proportion 
        shuffle=True,
        seed=pick_seed(**kwargs)
    )
    resplit_ratio = 100 * train_eval_test_ratios[1] / (train_eval_test_ratios[1] + train_eval_test_ratios[2])
    temp_dsd = out_dsd["test"].train_test_split(
        train_size = resplit_ratio / 100, 
        shuffle=True, 
        seed=pick_seed(**kwargs)
    )
    out_dsd["train-eval"] = temp_dsd["train"]
    out_dsd["test"] = temp_dsd["test"]
    return out_dsd

def tokenize_dataset_dict(row: dict, label2id: dict[str:int], tokenizer,  tokenization_parameters: dict) -> dict:
    row = row.copy()
    tokenized_entry = tokenize_string(row["TEXT"],tokenizer, tokenization_parameters)
    return {
        **row,
        **tokenized_entry,
        "labels": label2id[row["LABEL"]]
    }

def load_training_arguments(
        output_dir :str, 
        batch_size_device: int, 
        total_batch_size : int = 16, 
        **kwargs
    ) -> TrainingArguments:
    device = get_device()
    return TrainingArguments(
        bf16=True, # Faster training
        # Hyperparameters
        num_train_epochs = kwargs.get("n_epochs", 4),
        learning_rate = kwargs.get("learning_rate", 1e-5),
        weight_decay  = kwargs.get("weight_decay", 0.0),
        warmup_ratio  = kwargs.get("warmup_ratio", 0.05),
        # dropout = kwargs.get("dropout", 0.1), #TODO check for dropout
        # Second order hyperparameters
        per_device_train_batch_size = batch_size_device,
        per_device_eval_batch_size = batch_size_device,
        gradient_accumulation_steps = total_batch_size // batch_size_device,
        # Metrics
        metric_for_best_model="f1_macro",
        # Pipe
        output_dir = output_dir,
        overwrite_output_dir=True,
        eval_strategy = "epoch",
        logging_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        save_total_limit =  2,
        disable_tqdm = kwargs.get("disable_tqdm", False), 
        dataloader_pin_memory = False if device == "cuda" else True,
    )

def compute_metrics_multiclass(model_output: EvalPrediction):
    if isinstance(model_output.predictions,tuple):
        results_matrix = model_output.predictions[0]
    else:
        results_matrix = model_output.predictions
    y_true : list[int] = model_output.label_ids
    y_pred_probs = Tensor(results_matrix).softmax(1).numpy()
    y_pred = np.argmax(y_pred_probs, axis = 1).reshape(-1)

    return {
        "f1_macro": f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    }

def get_device() -> device:
    if cuda_available():
        empty_cache()
        return device("cuda")
    if mps_available():
        return device("mps")
    return device("cpu")
def clean():
    """
    """
    empty_cache()
    if cuda_available():
        synchronize()
        ipc_collect()
    gc_collect()
    print("Memory flushed")

def train_model(
    model, 
    training_args : TrainingArguments,
    dsd : DatasetDict,
    test_mode: bool = False, 
) -> None :
    """
    """
    print(f"Train_model call")
    try: 
        device = get_device()
        for split in dsd:
            dsd[split] = dsd[split].with_format("torch", device=device)
            if test_mode:
                dsd[split] = dsd[split].select(range(20))
        
        model = model.to(device=device)
        trainer = Trainer(
            model, 
            args = training_args,
            train_dataset=dsd["train"],
            eval_dataset=dsd["train-eval"], 
            compute_metrics = compute_metrics_multiclass,
        )
        print(f"Begin training on {device}")
        trainer.train()
        return trainer.state.best_model_checkpoint
    except Exception as e:
        print(f"ERROR in train_model: \n{e}")
    finally:
        del model, trainer, dsd
        clean()

def predict(model, ds : Dataset, batch_size: int, id2label: dict[int:str])->pd.DataFrame:
    print("PREDICT")
    if "input_ids" not in ds.features:
        raise ValueError("Please tokenize texts first")
    if not np.isin(["ID", "LABEL"], list(ds.features.keys())).all():
        raise ValueError("Please sanitize you Dataset first")
    
    device = get_device()
    print(f"Predict on {device}")
    
    ds = ds.with_format("torch", device=device)
    model = model.to(device=device)

    output_df = []
    for batch in ds.batch(batch_size):
        probs = model(input_ids = batch["input_ids"]).logits.detach().cpu().softmax(1).numpy()
        y_pred = np.argmax(probs, axis = 1).reshape(-1)

        output_df += [
            {
                "ID": id,
                "GS-LABEL": label,
                "PRED-LABEL": id2label[int(pred)],
            }
            for id, label, pred in zip(batch["ID"], batch["LABEL"], y_pred)
        ]
    return pd.DataFrame(output_df).set_index("ID")

def create_hash(**kwargs)->str:
    s = "|".join([f"{k}:{v}" for k,v in kwargs.items()])
    h = hashlib.new('sha256')
    h.update(s.encode())
    return h.hexdigest()


def prepare_environment():
    if not Path("./models").is_dir():
        os.mkdir("./models")
    if not Path("./predictions_save").is_dir():
        os.mkdir("./predictions_save")
    if not Path("./saving_logs.json").exists():
        with open("./saving_logs.json", "w") as file:
            json.dump({}, file)
    if not Path("./config.json").exists():
        raise ValueError("No config file, focus please.")