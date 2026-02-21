# IMPORTS ######################################################################
import os
from typing import Any

import pandas as pd

from ..CustomLogger import CustomLogger
from .TestOneEpoch import TestOneEpoch
# SCRIPTS ######################################################################
class TestAllEpochs:
    """TestAllEpochs is an object calling TestOneEpoch for all epochs in a
    folder.

    The main features and functions:
        - Evaluates the number of epochs for a given training
        - Run the TestOneEpoch routine for all epochs
        - Collect the results
        - Save the results

    The routine function proceeds to all these steps.   
    """
    def __init__(self, 
        foldername : str, 
        logger : CustomLogger, 
        device : str|None = None,
        batch_size : int = 64,
        ) -> None:
        """Builds the TestAllEpochs object.

        Parameters:
        -----------
            - foldername (str): full path to the checkpoints (saved during training)
                Equivalent to output_dir from CustomTransformersPipeline.
            - logger (CustomLogger): will give information as the data is processed.
            - device (str or None, default = None): device to load the model on.
            - batch_size (int, default = None): the batch size used during the 
                testing.
         
        Returns:
        --------
            /
        
        Initialised variables:
        ----------------------
            - self.__foldername (str): full path to the checkpoints (saved during 
                training)Equivalent to output_dir from CustomTransformersPipeline.
            - self.__n_epochs (int): number of epochs during the training.
            - self.__results (lict[dict[str:Any]]): list of the results to be saved.
            - self.__device (str or None, default = None): the batch size used
                during the testing.
            - self.__logger (CustomLogger): will give information as the data is 
                processed.
            - self.__batch_size (int): the batch size used during the testing.
        """
        self.__foldername : str = foldername
        self.__logger : CustomLogger = logger
        self.__n_epochs : int = len(
            [f for f in os.listdir(foldername) if f.startswith("checkpoint")])
        self.__scores : list[dict[str:Any]] = []
        self.__prediction_dfs : dict[str:pd.DataFrame] = {}
        self.__device : str|None = device
        self.__batch_size : int = batch_size

    def run_tests(self, foldername_data : str|None = None, additional_tags : dict = {}) -> None:
        """Run the TestOneEpoch routine for all epochs.

        Parameters:
        -----------
            - additional_tags(dict[str:Any]): Allows to include additionnal tags
                to all results.

        Returns:
        --------
            /
        """
        if foldername_data is not None:
            foldername_data_ = foldername_data
        else:
            foldername_data_ = self.__foldername + "/data"
            
        for epoch in range(1, self.__n_epochs + 1) :
            scores, prediction_df = TestOneEpoch(
                    foldername_model = self.__foldername, 
                    foldername_data = foldername_data_,
                    epoch = epoch, 
                    logger = self.__logger, 
                    device = self.__device, 
                    batch_size = self.__batch_size
                ).\
                routine(additional_tags)
            
            self.__scores.append(scores)
            self.__prediction_dfs[f"epoch_{epoch}"] = prediction_df
    
    def save_results(self, filename_scores : str, foldername_predictions : str) -> None:
        """Save the results in a csv file either by creating it or adding rows to 
        an existing file.

        Parameters:
        -----------
            - filename_scores (str): csv file to save scores into.
            - foldername_predictions (str): folder to save the predictions for each epoch

        Returns:
        --------
            /
        """
        if not os.path.isdir(foldername_predictions):
            os.makedirs(foldername_predictions)
        for epoch in self.__prediction_dfs:
            self.__prediction_dfs[epoch].to_csv(f"{foldername_predictions}/{epoch}.csv", index = False)
            
        try : 
            df = pd.read_csv(filename_scores)
            df = pd.concat((df, pd.DataFrame(self.__scores)))
        except:
            df = pd.DataFrame(self.__scores)
        finally:
            df.to_csv(filename_scores, index = False)
    
    def routine(self, 
        filename_scores : str, 
        foldername_predictions : str,
        foldername_data : str|None = None, 
        additional_tags : dict = {}
        ) -> None:
        """Routine used to run all the TestOneEpochs and save the results.

        The error catching is very coarse and only helps narrow down where the 
        routine stopped. Needs an upgrade.

        Parameters: 
        -----------
            - filename (str): csv file to save results into.
            - foldername_predictions (str): folder to save the predictions for each epoch
            - additional_tags(dict[str:Any]): Allows to include additionnal tags
                to all results.
        
        Returns:
        --------
            /
        """
        self.__logger(f"[TestAllEpochs] Routine start {self.__n_epochs} epochs---", 
            skip_line="before")
        
        self.run_tests(foldername_data, additional_tags)
        try:
            self.save_results(filename_scores, foldername_predictions) 
        except Exception as e:
            raise ValueError(f"Test All Epochs failed saving.\n\nError:\n{e}")
        
        self.__logger("[TestAllEpochs] Routine finish ---", skip_line="after")