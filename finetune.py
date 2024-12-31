"""
Multi-hypotheses-based Curriculum Learning for Semi-supervised Automatic Speech Recognition

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

-------------------------------------------------------------------------
File: finetune.py
 - finetune the seed model incorporating the unlabeled instances with pseudo labels
"""

import os
import tarfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from io import BytesIO
from urllib.request import urlopen

import torch
from torch import nn
from transformers import Wav2Vec2ForCTC
import torch.nn.functional as F

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss, score_to_weight, score_to_conf
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, WarmupCosineDecay, ReduceLROnPlateau
from mltu.augmentors import RandomAudioNoise, RandomAudioPitchShift, RandomAudioTimeStretch

from mltu.preprocessors import AudioReader
from mltu.transformers import LabelIndexer, LabelPadding, AudioPadding

from configs import ModelConfigs
from sklearn.model_selection import train_test_split

import time
import random
import librosa


def score_to_conf(x, temperature=1):
    if isinstance(x, float):
        x = [x]  # Convert single float to list
    x = np.array([float(x_) for x_ in x])
    e_x = np.exp(x / temperature)
    return e_x
configs = ModelConfigs()
torch.cuda.empty_cache()

# Read data
if configs.dataset == "LJSpeech-1.1":
    pretrained_path = "Models/10_wav2vec2_torch/202310231556/"
elif configs.dataset == "dev-clean":
    pretrained_path = "Models/10_wav2vec2_torch/202311082018/"
elif configs.dataset =="cv":
    pretrained_path = "Models/CV/202410221739/"
    dataset_path = "./Datasets/cv/en"
    metadata_path = dataset_path + "/other.tsv"
    wavs_path = dataset_path + "/wavclips/"
    # Read the TSV file with path and accent information
    metadata_df = pd.read_csv(metadata_path, sep='\t', usecols=['path', 'sentence', 'accents'])
    # Create a dictionary to map paths to accents
    path_to_accent = {row['path']: row['accents'] for _, row in metadata_df.iterrows()}

elif configs.dataset == "opensrl":
    pretrained_path = "Models/opensrl/202410221428/"
    target_accent = "southern_english_male"
    dataset_path = f"./Datasets/{configs.dataset}/"
    accent_list = [f.path.split('/')[-1] for f in os.scandir(dataset_path) if f.is_dir()]
    for accent in accent_list:
        metadata_path = dataset_path + accent+ "/line_index.csv"
        metadata_df = pd.read_csv(metadata_path, sep=',')

unlabeled_dataset = pd.read_csv(pretrained_path + "pred10.csv").values.tolist()
unlabeled_dataset_withlabel = pd.read_csv(pretrained_path + "val.csv").values.tolist()
labeled_dataset = pd.read_csv(pretrained_path + "train.csv").values.tolist()
labeled_dataset_pred = pd.read_csv(pretrained_path + "train_pred.csv").values.tolist()

target_english_instances = []
target_conf = 0
for data in labeled_dataset_pred:
    target_conf += score_to_conf(data[2])
    target_english_instances.append(data[0])
target_conf /= len(labeled_dataset)

dataset, test_dataset = [], []

vocab = [' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# target English 인스턴스 분류
remaining_unlabeled_dataset = []
target_indices = []  # 인덱스를 저장할 리스트
for idx, data in enumerate(unlabeled_dataset):
    if configs.dataset =='opensrl':
        target_accent = "southern_english_male"
        filepath = data[0]  # 파일 이름 추출
        accent = filepath.split('/')[-2]
    if os.path.exists(filepath):
        if accent == target_accent:
            target_indices.append(idx)  # 해당 인덱스를 저장
            target_english_instances.append(data[0])
        else:
            remaining_unlabeled_dataset.append(data)

# target English 인스턴스를 80:20으로 분리
train_target_indices, test_target_indices = train_test_split(target_indices, test_size=0.75, random_state=42)

# 인덱스를 통해 train_target 데이터와 test_target 데이터를 구성
train_target_data = [unlabeled_dataset[idx] for idx in train_target_indices]
test_target_data = [unlabeled_dataset_withlabel[idx] for idx in test_target_indices]

# 80%의 target English 인스턴스를 다시 unlabeled_dataset에 추가
unlabeled_dataset = train_target_data + remaining_unlabeled_dataset
# unlabeled_dataset = train_target_data

# 20%의 target English 인스턴스를 test_dataset에 추가
test_dataset.extend(test_target_data)
hardnesses = []
ratio = configs.train_ratio

over = 0
less = 0
# for data in unlabeled_dataset[:int(len(unlabeled_dataset)*ratio)]:
print(f"Total train instances : {len(unlabeled_dataset)}")
print(f"target instances : {len(train_target_data)}")
print(f"Non-target instances : {len(unlabeled_dataset)-len(train_target_data)}")

for data in unlabeled_dataset:
    audio, _ = librosa.load(data[0], sr=8000)
    pred_labels = data[1].split('_')
    scores = data[2].split('_')
    weights = score_to_weight(scores)
    conf = score_to_conf(scores) / target_conf
    nontarget_less = 0
    nontarget_over = 0
    if configs.candidate == True:
        # k = configs.top_k
        if data[0] in target_english_instances:
            k = configs.top_k
        else:
            k = int(configs.top_k * max(conf))
            if k < 1:
                k = 1
                nontarget_less += 1
            elif k > configs.top_k : 
                k = configs.top_k
                nontarget_over += 1
        for i in range(k):
            idx = random.choices(range(len(pred_labels)), weights=weights)[0]
            # idx = random.choices(range(len(pred_labels)))[0]
            selected_element = pred_labels[idx]
            if (selected_element.replace(' ','')):
                if configs.curriculum == 'speed':
                    hardness = len(selected_element)/len(audio) * len(selected_element)
                elif configs.curriculum == 'confidence':
                    hardness = len(selected_element) + weights[idx]
                hardnesses.append(hardness)
                dataset.append([data[0], selected_element])            
    else:
        pred_labels = data[1].split('_')[0]
        dataset.append([data[0], pred_labels])
        if configs.curriculum == 'speed':
            hardness = len(pred_labels) / len(audio) * len(pred_labels)
        elif configs.curriculum == 'confidence':
            hardness = len(pred_labels) + 1
        hardnesses.append(hardness)

print(f"Number of Sample: {len(dataset)-configs.top_k*len(train_target_data)}")
print(f"Non-target less : {nontarget_less} || over: {nontarget_over}")

pred_dict = {pred_data[0]: score_to_conf(pred_data[2]) for pred_data in labeled_dataset_pred}

for data in labeled_dataset:
    audio, _ = librosa.load(data[0], sr=8000)
    if configs.curriculum == 'speed':
        hardness = len(data[1])/len(audio) * len(data[1])
    elif configs.curriculum == 'confidence':
        hardness = len(data[1]) + 1
    hardnesses.append(hardness)
    dataset.append(data)


# for data in unlabeled_dataset_withlabel[int(len(unlabeled_dataset)*ratio):]:
#     test_dataset.append(data)

if configs.sort == True:
    sorted_idx = np.argsort(hardnesses)
    dataset_ordered = [dataset[i] for i in sorted_idx]

else: dataset_ordered = dataset

# Create a data provider for the dataset
train_dataProvider = DataProvider(
    dataset=dataset_ordered,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=8000),
        ],
    transformers=[
        LabelIndexer(vocab),
        ],
    use_cache=False,
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True),
        LabelPadding(padding_value=len(vocab), use_on_batch=True),
    ],
    use_multiprocessing=True,
    max_queue_size=10,
    workers=configs.train_workers,
    shuffle=False,
)

test_dataProvider = DataProvider(
    dataset=test_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=8000),
        ],
    transformers=[
        LabelIndexer(vocab),
        ],
    use_cache=False,
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True),
        LabelPadding(padding_value=len(vocab), use_on_batch=True),
    ],
    use_multiprocessing=True,
    max_queue_size=10,
    workers=configs.train_workers,
)

vocab = sorted(vocab)
configs.vocab = vocab
configs.save()

class CustomWav2Vec2Model(nn.Module):
    def __init__(self, hidden_states, dropout_rate=0.2, **kwargs):
        super(CustomWav2Vec2Model, self).__init__( **kwargs)

        pretrained_name = "facebook/wav2vec2-base-960h"
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_name, vocab_size=hidden_states, ignore_mismatched_sizes=True)
        self.model.freeze_feature_encoder() # this part does not need to be fine-tuned

    def forward(self, inputs):
        output = self.model(inputs, attention_mask=None).logits
        # Apply softmax
        output = F.log_softmax(output, -1)
        return output

custom_model = CustomWav2Vec2Model(hidden_states = len(vocab)+1)
custom_model.load_state_dict(torch.load(pretrained_path + '/model.pt'))

# put on cuda device if available
if torch.cuda.is_available():
    custom_model = custom_model.to(f"cuda:{configs.device}")

# create callbacks
warmupCosineDecay = WarmupCosineDecay(
    lr_after_warmup=configs.lr_after_warmup,
    warmup_epochs=configs.warmup_epochs,
    decay_epochs=configs.decay_epochs,
    final_lr=configs.final_lr,
    initial_lr=configs.init_lr,
    verbose=True,
)

tb_callback = TensorBoard(configs.model_path + "/logs")
earlyStopping = EarlyStopping(monitor="val_CER", patience=60, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.max_audio_length),
    verbose=1,
    metadata={"vocab": configs.vocab},
    dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size", 1: "sequence_length"}}
)

# create model object that will handle training and testing of the network
model = Model(
    custom_model,
    loss = CTCLoss(blank=len(configs.vocab), zero_infinity=True),
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=configs.init_lr, weight_decay=configs.weight_decay),
    metrics=[
        CERMetric(configs.vocab),
        WERMetric(configs.vocab)
    ],
    mixed_precision=configs.mixed_precision,
)

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, f"{configs.curriculum}_{configs.top_k}_train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, f"{configs.curriculum}_{configs.top_k}_val.csv"))

model.fit(
    train_dataProvider,
    test_dataProvider,
    epochs=configs.train_epochs,
    callbacks=[
        warmupCosineDecay,
        tb_callback,
        earlyStopping,
        modelCheckpoint,
        model2onnx
    ]
)