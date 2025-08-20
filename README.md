# Accurate Semi-supervised Automatic Speech Recognition via Multi-hypotheses-based Curriculum Learning

This is the code repository for Multi-hypotheses-based Curriculum Learning for Semi-supervised Automatic Speech Recognition.
This includes the implementation of MOCA (**M**ulti-hyp**O**theses-based **C**urriculum learning for semi-supervised **A**SR), and MOCA-S
our novel approach for  semi-supervised automatic speech recognition (ASR) for ordinary and characterized speech, respectively.

## Abstract
How can we build accurate transcription models for both ordinary speech and characterized speech in a semi-supervised setting? ASR (Automatic Speech Recognition) systems are widely used in various real-world applications, including translation systems and transcription services. ASR models are tailored to serve one of two types of speeches: 1) ordinary speech (e.g., speeches from the general population) and 2) characterized speech (e.g., speeches from speakers with special traits, such as certain nationalities or speech disorders). Recently, the limited availability of labeled speech data and the high cost of manual labeling have drawn significant attention to the development of semi-supervised ASR systems. Previous semi-supervised ASR models employ a pseudo-labeling scheme to incorporate unlabeled examples during training. However, these methods rely heavily on pseudo labels during training and are therefore highly sensitive to the quality of pseudo labels. The issue of low-quality pseudo labels is particularly pronounced for characterized speech, due to the limited availability of data specific to a certain trait. This scarcity hinders the initial ASR model’s ability to effectively capture the unique characteristics of characterized speech, resulting in inaccurate pseudo labels.
In this paper, we propose a framework for training accurate ASR models for both ordinary and characterized speeches in a semi-supervised setting. Specifically, we propose MOCA (Multi-hypotheses-based Curriculum learning for semi-supervised ASR) for ordinary speech and MOCA-S for characterized speech. MOCA and MOCA-S generate multiple hypotheses for each speech instance to reduce the heavy reliance on potentially inaccurate pseudo labels. Moreover, MOCA-S for characterized speech effectively supplements the limited trait-specific speech data by exploiting speeches of the other traits. Specifically, MOCA-S adjusts the number of pseudo labels based on the relevance to the target trait. Extensive experiments on real-world speech datasets show that MOCA and MOCA-S significantly improve the accuracy of previous ASR models.

## Requirements

We recommend using the following versions of packages:
- `PyYAML>=6.0`
- `tqdm`
- `pandas`
- `numpy`
- `opencv-python`
- `Pillow>=9.4.0`
- `onnxruntime>=1.15.0`

## Data Overview
The datasets are available at (https://drive.google.com/drive/folders/1RSyw6aExar_5Li_j2Jy59q_IfErLkNH1?usp=share_link).

|        **Dataset**        |                  **Path**                   | 
|:-------------------------:|:-------------------------------------------:| 
|       **LJSpeech**        |           `Datasets/LJSpeech-1.1`           | 

## How to Run
You can run the demo script in the directory by the following code.
```
python finetune.py
```
The demo script finetunes a seed model incorporating the unlabeled examples with multiple pseudo labels.
You can reproduce the results in the paper by running the demo script while changing the configuration file (`./configs.py`).

## References
The codes are written based on the `mltu` package (https://github.com/pythonlessons/mltu).

## Reference
If you use this code, please cite the following paper.
```shell
@inproceedings{kim2024accurate,
  title={Accurate Semi-supervised Automatic Speech Recognition via Multi-hypotheses-Based Curriculum Learning},
  author={Kim, Junghun and Park, Ka Hyun and Kang, U},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={40--52},
  year={2024},
  organization={Springer}
}
```
