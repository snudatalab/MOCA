import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
# Define the score_to_conf function
def score_to_conf(x, temperature=10):
    if isinstance(x, float):
        x = [x]  # Convert single float to list
    x = np.array([float(x_) for x_ in x])
    e_x = np.exp(x / temperature)
    return e_x

# Define file paths
path_to_accent={}
pretrained_path = "Models/opensrl/202410221428/"
target_accent = "som"
dataset_path = f"./Datasets/opensrl/"
accent_list = ['som','scm','nom','mim','irm','wem']

unlabeled_pred = pd.read_csv(pretrained_path + "pred10.csv").values.tolist()
unlabeled_dataset_withlabel = pd.read_csv(pretrained_path + "val.csv").values.tolist()
labeled_dataset = pd.read_csv(pretrained_path + "train.csv").values.tolist()
labeled_dataset_pred = pd.read_csv(pretrained_path + "train_pred.csv").values.tolist()

# Load metadata
# metadata_df = pd.read_csv(metadata_path, sep='\t', usecols=['path', 'sentence', 'accents'])
# path_to_accent = {row['path']: row['accents'] for _, row in metadata_df.iterrows()}

# Calculate target labeled confidence
target_labeled_conf = 0
count_target_accent = 0
for data in labeled_dataset_pred:
    filename = data[0].split('/')[-1]
    accent = filename.split('_')[0]
    if accent == target_accent:
        target_labeled_conf += score_to_conf(data[2])[0]  # Assume score_to_conf returns an array
        count_target_accent += 1

# Avoid division by zero
if count_target_accent > 0:
    target_labeled_conf /= count_target_accent
# Accumulate confidence values by accent
accent_to_weight = {}
for data in unlabeled_pred:
    filename = data[0].split('/')[-1][:-4]
    conf = score_to_conf(float(data[2].split('_')[0]))[0]  # Assume score_to_conf returns an array
    accent = filename.split('_')[0]
    if accent:
        normalized_conf = conf / target_labeled_conf
        if accent in accent_to_weight:
            accent_to_weight[accent].append(normalized_conf)
        else:
            accent_to_weight[accent] = [normalized_conf]
print(accent_to_weight)
# Plot KDEs for accents with more than 100 instances
plt.figure(figsize=(12, 8))
for accent, conf_values in accent_to_weight.items():
    # if len(conf_values) > 100 and accent in ['United States English', 'Canadian English', 'Australian English', 'Filipino']:
    if len(conf_values) > 100:
        sns.kdeplot(conf_values, shade=True, label=f"{accent} (Mean : {np.mean(conf_values):.2f})")


# Customize plot
plt.title("Distributions of Normalized Likelihoods", fontsize=30)
plt.xlabel("Normalized Likelihoods", fontsize=30)
plt.ylabel("Density", fontsize=30)
plt.xlim(0, 2)  # Set x-axis from 0 to 3
plt.legend(fontsize=24)

# Adjust layout and save the plot with better padding
plt.tight_layout()
output_path = "confidence_distributions_by_accent_4.png"
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')  # Save at 300 DPI for high quality
plt.show()
