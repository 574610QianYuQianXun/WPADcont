import math
from typing import List, Any, Dict
import torch
import logging
import os

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FedAvg:

    def __init__(self, args) -> None:
        self.args = args

    # FedAvg aggregation
    def aggr(self,clients_param):
        sum_model = 0
        for _id, model_param in clients_param.items():
            sum_model += model_param
        # print(sum_model / len(clients_param))
        return sum_model / len(clients_param)

    def accumulate_weights(self, weight_accumulator, local_update):
        for name, value in local_update.items():
            weight_accumulator[name].add_(value)