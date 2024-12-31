"""
Multi-hypotheses-based Curriculum Learning for Semi-supervised Automatic Speech Recognition

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

-------------------------------------------------------------------------
File: test.py
 - test the trained model on a test dataset
"""

import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import my_ctc_decoder, get_cer, get_wer, beam_decoder

from configs import ModelConfigs

configs = ModelConfigs()

class Wav2vec2(OnnxInferenceModel):
    """
    wav2vec model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, audio: np.ndarray):
        audio = np.expand_dims(audio, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: audio})[0]

        token = self.metadata["vocab"]
        for toc in ['-', '|']:
            if toc not in token:
                token.append(toc)
        texts, scores = beam_decoder(preds, None, token)

        return texts, scores


if __name__ == "__main__":
    import librosa
    import pandas as pd
    from tqdm import tqdm

    if configs.dataset == "LJSpeech-1.1":
        save_path = "Models/10_wav2vec2_torch/202310231556/"
    elif configs.dataset == "dev-clean":
        save_path = "Models/10_wav2vec2_torch/202311082018/"
    elif configs.dataset == "cv":
        # save_path ="Models/CV/202410052314/"
        # save_path ="Models/CV/202410081608/"
        ## len(vocab)+1
        # save_path ="Models/CV/202410111406/"
        save_path = "Models/CV/202410221428/"
    elif configs.dataset == "opensrl":
        save_path = "Models/opensrl/202410221428/"

    model = Wav2vec2(model_path = save_path + "model.onnx")
    val_dataset = pd.read_csv(save_path + "val.csv").values.tolist()
    src_dataset = pd.read_csv(save_path + "src.csv").values.tolist()

    # The list of multiple [audio_path, label] for validation
    accum_cer, accum_wer = [], []
    accum_labels, accum_scores, accum_path = [], [], []

    pbar = tqdm(val_dataset + src_dataset)

    for vaw_path, label in pbar:
        audio, sr = librosa.load(vaw_path, sr=8000)

        # best_prediction_text = model.predict(audio)
        prediction_texts, prediction_scores = model.predict(audio)
        # prediction_text = prediction_texts[0]
        best_prediction_text = prediction_texts[0]
        prediction_text = "_".join([text for text in prediction_texts])

        prediction_score = "_".join([str(score) for score in prediction_scores])
        # prediction_score = prediction_scores[0]
        accum_labels.append(prediction_text)
        accum_path.append(vaw_path)
        accum_scores.append(prediction_score)

        cer = get_cer(best_prediction_text, label)
        wer = get_wer(best_prediction_text, label)

        accum_cer.append(cer)
        accum_wer.append(wer)

        pbar.set_description(f"Average CER: {np.average(accum_cer):.4f}, Average WER: {np.average(accum_wer):.4f}")

    df = pd.DataFrame({"0": accum_path, "1": accum_labels, "2": accum_scores})
    df.to_csv(save_path + "pred10.csv", index=False)

    train_dataset = pd.read_csv(save_path + "train.csv").values.tolist()
    
    # The list of multiple [audio_path, label] for validation
    accum_cer, accum_wer = [], []
    accum_labels, accum_scores, accum_path = [], [], []

    pbar = tqdm(train_dataset)
    for vaw_path, label in pbar:
        audio, sr = librosa.load(vaw_path, sr=8000)

        prediction_texts, prediction_scores = model.predict(audio)
        prediction_text = prediction_texts[0]
        best_prediction_text = prediction_texts[0]

        # prediction_score = "_".join([str(score) for score in prediction_scores])
        prediction_score = prediction_scores[0]
        accum_labels.append(prediction_text)
        accum_path.append(vaw_path)
        accum_scores.append(prediction_score)

        cer = get_cer(best_prediction_text, label)
        wer = get_wer(best_prediction_text, label)

        accum_cer.append(cer)
        accum_wer.append(wer)

        pbar.set_description(f"Average CER: {np.average(accum_cer):.4f}, Average WER: {np.average(accum_wer):.4f}")

    train_df = pd.DataFrame({"0": accum_path, "1": accum_labels, "2": accum_scores})
    train_df.to_csv(save_path + "train_pred.csv", index=False)
