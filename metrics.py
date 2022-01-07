import numpy as np
import json
import shutil
from model import ModelVad
from dataset import AVADataset
from training.prepare import create_train_test_dataset, create_spec_dataset_only_speech
from training.train_metrics import test_result, plot_PR_ROC
from torch.utils.data import DataLoader
import torch
import sklearn.metrics as metrics

if __name__ == '__main__':

    with open('config.json', 'r') as f:
        config = json.load(f)

    model = ModelVad()
    model.load_state_dict(torch.load(config['path_to_checkpoint']))
    metric = metrics.roc_auc_score
    criterion = torch.nn.BCELoss()

    labels = create_spec_dataset_only_speech(config['speech_labels_path'], config['path_to_save_specs'], config['delimiter'])

    np.save(config['path_to_save_specs'] + '/labels', labels)

    _, test_df = create_train_test_dataset(labels, 1)

    mean = -3.052366018295288
    std = 2.4621522426605225

    test_dataset = AVADataset(test_df, mean, std)

    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

    y_true, y_proba = test_result(model, test_data_loader, criterion, metric)

    plot_PR_ROC(y_true, y_proba)

    if config['is_save_specs'] != '1':
        shutil.rmtree(config['path_to_save_specs'])