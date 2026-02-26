# IMPORTS ######################################################################
from pathlib import Path
from typing import Any

from datasets import load_from_disk, Dataset, DatasetDict
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score
from torch import Tensor, load, no_grad
from torch.cuda import is_available as cuda_available
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from ..general import checkpoint_to_load, clean
from ..CustomLogger import CustomLogger
# SCRIPTS ######################################################################
class TestOneEpoch: 
    """TestOneEpoch is an object that loads a model after n epochs of training 
    and evaluate it's performances. It is meant to work with the files created during 
    the CustomTransformersPipeline routine.

    The main features and functions are:
        - Load a model after n epochs of training.
        - Load encoded data and labels (test set)
        - Classify entries and evaluate performances
        - Return a dictionnary resuming the performances and metadata describing
        the training.

    The routine function proceeds to all these steps.   
    """
    # UPGRADE make possible to change the measure
    def __init__(self, 
        foldername_model: str, 
        epoch : int, 
        logger : CustomLogger,
        foldername_data: str | None = None,
        device : str|None = None, 
        batch_size : int = 64
        ) -> None:
        """Builds the TestOneEpoch object.

        Possible UPDATE: change the measure to use.

        Parameters:
        -----------
            - foldername_model (str): full path to the checkpoints (saved during 
                training) Equivalent to output_dir from CustomTransformersPipeline.
            - epoch (int): number of epochs of training, will choose what checkpoint
                to load. Epoch 0 = no training.
            - logger (CustomLogger): will give information as the data is processed.
            - foldername_data (str|None): full path to a dataset. Must be loadable with 
                datasets.load_from_disk. If None, will use the foldername model
            - device (str or None, default = None): device to load the model on.
                can be 'cpu', 'cuda' or 'cuda:X'.
            - batch_size (int, default = 64): the batch size used during the testing.

        Returns:
        --------
            /
        
        Inisialised variables:
        ----------------------
            DATA 
            - self.__ds (DatasetDict): dataset created during the step 1. Contains 
                at least a "test" split and 3 columns ('input_ids', 'attention_mask'
                and 'labels')

            MODEL
            - self.__foldername (str): full path to the checkpoints (saved during 
                training)Equivalent to output_dir from CustomTransformersPipeline.
            - self.__epoch (int): number of epochs of training, will choose what 
                checkpoint to load. Epoch 0 = no training.
            - self.__checkpoint (str): checkpoint in the folder corresponding to 
                the epoch we want to test.
            - self.__model (AutoModelForSequenceClassification): model loaded.
            - device (str or None, default = None): device to load the model on, 
                can be 'cpu', 'cuda' or 'cuda:X'.
            - self.__batch_size (int): the batch size used during the testing.
            
            METADATA 
            - self.__training_args (TrainingArguments): training arguments used 
                during the training to feed the metadata.
            - self.__model_name (str): name of the model used to feed the metadata.
            - self.__metric (str): the metric used to evaluate the performance of
                the model. For now only f1_macro available. The metric will be 
                returned in the metadata.
            - self.__score (float): the score of the model after testing.

            COMMUNICATION AND SECURITY
            - self.__logger (CustomLogger): will give information as the data is 
                processed.
        """
        self.__foldername_model : str = foldername_model
        self.__epoch : str = epoch
        self.__logger : CustomLogger = logger
        if foldername_data: 
            self.__ds : DatasetDict = load_from_disk(foldername_data)
        else: 
            self.__ds : DatasetDict = load_from_disk(Path(foldername_model).joinpath("data"))
        self.__batch_size : int = batch_size

        if device is None : 
            self.device = "cuda" if cuda_available() else "cpu"
        else :
            self.device = device

        # Choose what checkpoint to load, and load the model
        self.__checkpoint : str = checkpoint_to_load(foldername_model, epoch)
        self.__model = AutoModelForSequenceClassification.\
            from_pretrained(f"{foldername_model}/{self.__checkpoint}").\
            to(device = self.device)

        # Load the training args to retrieve return afterwards.
        self.__training_args : TrainingArguments = load(
            f"{foldername_model}/{self.__checkpoint}/training_args.bin", 
            weights_only=False
        )
        
        # Load the model name
        with open(f"{foldername_model}/model_name.txt", "r") as file:
            self.__model_name : str = file.read()

        self.__metric : str = "f1_macro"

        (self.__score, self.__prediction_df) = (None, ) * 2

    def run_test(self):
        """In batch, predicts the label and keep in memory the true label and 
        performance evaluation. 

        Parameters:
        -----------
            /
        
        Returns:
        --------
        """
        index_element : list[int|str] = []
        labels_true : list[int] = []
        labels_pred : list[int] = []
        with no_grad():
            for batch in self.__ds["test"].batch(self.__batch_size, drop_last_batch=False):
                model_input = {
                    'input_ids' : Tensor(batch['input_ids']).\
                                    to(dtype = int).\
                                    to(device=self.device), 
                    'attention_mask' : Tensor(batch['attention_mask']).\
                                    to(dtype = int).\
                                    to(device=self.device) 
                }

                logits : np.ndarray = self.__model(**model_input).logits.\
                    detach().cpu().numpy()
                
                batch_of_true_label : list[int] = [
                    np.argmax(row).item() for row in batch["labels"]]
                labels_true.extend(batch_of_true_label)

                batch_of_pred_label : list[int] = [
                    np.argmax(row).item() for row in logits]
                labels_pred.extend(batch_of_pred_label)
                
                batch_of_indexes = [str(idx) for idx in batch["ID"]]
                index_element.extend(batch_of_indexes)
        
        # Evaluate performance`
        self.__prediction_df = pd.DataFrame({
            "ID": index_element, 
            "LABEL-GS": labels_true, 
            "LABEL-PRED": labels_pred,
        })
        self.__score = f1_score(labels_true, labels_pred, average='macro')

        # Logging
        self.__logger((f"(Epoch {self.__epoch} - checkpoint {self.__checkpoint}) "
                       f"Testing - Done (score : {self.__score})"))
        
    def return_result(self, additional_tags : dict[str:Any] = {}) -> dict:
        """Builds a dictionnary mixing the results and metadata from the training
        routine. Allows for extra tags.

        Parameters:
        -----------
            - additional_tags(dict[str:Any]): Allows to include additionnal tags.
        
        Returns:
        --------
            - dict[str:Any]
        """
        return {
            "folder" : self.__foldername_model,
            "epoch" : int(self.__epoch) + 1,
            "score" : self.__score, 
            "measure" : self.__metric, 
            "learning_rate" : self.__training_args.learning_rate,
            "optim" : self.__training_args.optim,
            "warmup_ratio" : self.__training_args.warmup_ratio,
            "weight_decay" : self.__training_args.weight_decay,
            "embedding_model" : self.__model_name,
            **additional_tags
        }

    def routine(self, additional_tags : dict = {}) -> tuple[dict, pd.DataFrame]:
        """Routine used to load the models, run the testing and return the 
        metadata. 

        The error catching is very coarse and only helps narrow down where the 
        routine stopped. Needs an upgrade.
        
        Parameters:
        -----------
            - additional_tags(dict[str:Any]): Allows to include additionnal tags.
        
        Returns:
        --------
            - dict[str:Any]
            - pd.DataFrame
        """
        try: 
            self.run_test()
        except Exception as e:
            raise ValueError(f"Test One Epoch could not run the test.\n\nError:\n{e}")
        scores = self.return_result(additional_tags)
        del self.__model, self.__ds
        clean()
        return scores, self.__prediction_df
    
