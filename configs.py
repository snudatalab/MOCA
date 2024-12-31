"""
Multi-hypotheses-based Curriculum Learning for Semi-supervised Automatic Speech Recognition

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

-------------------------------------------------------------------------
File: configs.py
 - configuration for training
"""


import os
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        # self.model_path = os.path.join(
        #     "Models/10_wav2vec2_torch",
        #     datetime.strftime(datetime.now(), "%Y%m%d%H%M"),
        # )
        self.model_path = os.path.join(
            "Models/opensrl",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M"),
        )
        # self.dataset = "LJSpeech-1.1"
        # self.dataset = "cv"
        # self.dataset = "dev-clean"
        self.dataset = "opensrl"

        # self.dataset = "svarah"
        self.batch_size = 1
        self.train_epochs = 20
        self.train_workers = 1



        # self.init_lr = 1.0e-6
        # self.lr_after_warmup = 1e-04
        # self.final_lr = 5e-04

        # self.init_lr = 1.0e-7
        # self.lr_after_warmup = 1e-05
        # self.final_lr = 5e-05

        self.init_lr = 1.0e-8
        self.lr_after_warmup = 1e-06
        self.final_lr = 5e-06

        # self.init_lr = 1.0e-9
        # self.lr_after_warmup = 1e-07
        # self.final_lr = 5e-07
        
        # self.init_lr = 1.0e-10
        # self.lr_after_warmup = 1e-08
        # self.final_lr = 5e-08

        # self.lr_after_warmup = 1e-06
        # self.final_lr = 5e-05
        
        self.warmup_epochs = 10
        self.decay_epochs = 40
        self.weight_decay = 0.005
        self.mixed_precision = True

        self.max_audio_length = 246000
        self.max_label_length = 256

        self.top_k = 10
        self.candidate = True
        self.sort = False
        self.curriculum = "speed"
        # self.curriculum = "confidence"
        self.train_ratio = 4/9
        self.device = 0
        self.vocab = [' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']