"""
Multi-hypotheses-based Curriculum Learning for Semi-supervised Automatic Speech Recognition

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

-------------------------------------------------------------------------
File: train.py
 - train a seed model with a few labeled instances
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
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, WarmupCosineDecay
from mltu.augmentors import RandomAudioNoise, RandomAudioPitchShift, RandomAudioTimeStretch

from mltu.preprocessors import AudioReader
from mltu.transformers import LabelIndexer, LabelPadding, AudioPadding

from configs import ModelConfigs
import time
configs = ModelConfigs()
def download_and_unzip(url, extract_to="Datasets", chunk_size=1024*1024):
    """
    download dataset from "url", and extract to "extract_to" directory
    """
    http_response = urlopen(url)

    data = b""
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    # tarFile = tarfile.open(fileobj=BytesIO(data), mode="r|bz2")
    tarFile = tarfile.open(fileobj=BytesIO(data))
    tarFile.extractall(path=extract_to)
    tarFile.close()

vocab = [' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
             't', 'u', 'v', 'w', 'x', 'y', 'z']

dataset_name = configs.dataset
dataset = []
src_dataset = []

if dataset_name == "LJSpeech-1.1":
    dataset_path = os.path.join("Datasets", "LJSpeech-1.1")
    if not os.path.exists(dataset_path):
        download_and_unzip("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", extract_to="Datasets")

    metadata_path = dataset_path + "/metadata.csv"
    wavs_path = dataset_path + "/wavs/"

    # Read metadata file and parse it
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    for file_name, transcription, normalized_transcription in metadata_df.values.tolist():
        path = f"Datasets/LJSpeech-1.1/wavs/{file_name}.wav"
        new_label = "".join([l for l in normalized_transcription.lower() if l in vocab])
        dataset.append([path, new_label])


elif dataset_name in ['dev-clean', 'dev-other', 'test-clean', 'test-other']:
    dataset_path = os.path.join("Datasets", dataset_name)
    if not os.path.exists(dataset_path):
        data_link = "https://us.openslr.org/resources/12/" + dataset_name + ".tar.gz"
        download_and_unzip(data_link, extract_to=dataset_path)

    # Read data
    dataset_path = os.path.join(dataset_path, "LibriSpeech", dataset_name)
    for path1 in sorted(os.listdir(dataset_path)):
        for path2 in sorted(os.listdir(os.path.join(dataset_path, path1))):
            data_df = pd.read_csv(os.path.join(dataset_path, path1, path2, path1 + '-' + path2 + '.trans.txt'), header=None)
            data_list = np.array(data_df).squeeze()
            for instance in data_list:
                file_name = instance.split(" ")[0]
                path = os.path.join(dataset_path, path1, path2, file_name) + '.flac'
                upper_label = instance.split(" ")[1:]
                upper_label = " ".join(upper_label)
                new_label = "".join([l for l in upper_label.lower() if l in vocab])
                dataset.append([path, new_label])

elif dataset_name == "cv":
    dataset_path = "./Datasets/cv/en"
    # if not os.path.exists(dataset_path):
    #     download_and_unzip("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", extract_to="Datasets")
    metadata_path = dataset_path + "/other.tsv"
    wavs_path = dataset_path + "/wavclips/"
    # Read metadata file and parse it
    metadata_df = pd.read_csv(metadata_path, sep='\t', usecols=['path','sentence','accents'])
    for file_name, transcription, accent in metadata_df.values.tolist():
        filepath = f"./Datasets/cv/en/wavclips/{file_name}"

        if os.path.exists(filepath):
            new_label = "".join([l for l in transcription.lower() if l in vocab])
            if len(new_label.replace(' ',''))!=0 : 
                if accent == 'Scottish English':    
                    dataset.append([filepath, new_label])
                else :
                    src_dataset.append([filepath, new_label])


elif dataset_name == "opensrl":
    target_accent = "southern_english_male"
    dataset_path = f"./Datasets/{dataset_name}/"
    accent_list = [f.path.split('/')[-1] for f in os.scandir(dataset_path) if f.is_dir()]
    print(accent_list)
    for accent in accent_list:
        metadata_path = dataset_path+accent+ "/line_index.csv"
        metadata_df = pd.read_csv(metadata_path, sep=',')
        for _, file_name, transcription in metadata_df.values.tolist():
            file_name = file_name.strip()
            transcription = transcription.strip()
            filepath = f"{dataset_path}{accent}/{file_name}.wav"
            print(filepath)
            if os.path.exists(filepath):
                new_label = "".join([l for l in transcription.lower() if l in vocab])
                if len(new_label.replace(' ',''))!=0 : 
                    if accent == target_accent:    
                        dataset.append([filepath, new_label])
                    else :
                        src_dataset.append([filepath, new_label])
else:
    print("There is no such dataset")

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
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


train_dataProvider, test_dataProvider = data_provider.split(split=0.5)
# Create a data provider for the dataset
src_dataProvider = DataProvider(
    dataset=src_dataset,
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
    """
    our ASR structure that uses pre-trained wav2vec as the initial weights
    """
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
# custom_model = CustomWav2Vec2Model(hidden_states = len(vocab)+4)

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
earlyStopping = EarlyStopping(monitor="val_CER", patience=100, mode="min", verbose=1)
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
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))
src_dataProvider.to_csv(os.path.join(configs.model_path, "src.csv"))

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