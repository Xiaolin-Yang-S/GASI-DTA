import os
import json, torch

import numpy as np
import torch.backends.cudnn
import torch.utils.data


from model import (Predictor, GASIDTA)
from metrics import model_evaluate

from utils import argparser, DTADataset, GraphDataset, collate, predicting, read_data, train, setup_seed
import warnings

warnings.filterwarnings('ignore')


def create_dataset_for_train_test(affinity, dataset, fold):
    # load dataset
    dataset_path = 'data/' + dataset + '/'

    train_fold_origin = json.load(open(dataset_path + 'train_set.txt'))
    train_folds = []
    for i in range(len(train_fold_origin)):
        if i != fold:
            train_folds += train_fold_origin[i]
    test_fold = json.load(open(dataset_path + 'test_set.txt')) if fold == -100 else train_fold_origin[fold]


    rows, cols = np.where(np.isnan(affinity) == False)
    train_rows, train_cols = rows[train_folds], cols[train_folds]

    train_Y = affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

    test_rows, test_cols = rows[test_fold], cols[test_fold]
    test_Y = affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    return train_dataset, test_dataset


def train_test():
    FLAGS = argparser()

    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    cross_test = FLAGS.cross_test
    # NUM_EPOCHS = FLAGS.num_epochs
    NUM_EPOCHS = 2000
    LR = FLAGS.lr

    Architecture = GASIDTA

    fold = FLAGS.fold

    dataset = 'davis'
    cuda_name = f'cuda:1'

    model_name = Architecture.__name__
    if not FLAGS.weighted:
        model_name += "-noweight"
    if fold != -100:
        model_name += f"-{fold}"

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Epochs:", NUM_EPOCHS)
    print('batch size', TRAIN_BATCH_SIZE)
    print("Learning rate:", LR)
    print("Fold", fold)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)

    if os.path.exists(f"models/architecture/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/predictor/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/architecture/{dataset}/S1/test/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/test/")
    if os.path.exists(f"models/predictor/{dataset}/S1/test/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/test/")

    print("\ncreate dataset ......")

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    affinity = read_data(dataset)

    train_data, test_data = create_dataset_for_train_test(affinity, dataset, fold)

    print("create train_loader and test_loader ...")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print("create drug_graphs_dict and target_graphs_dict ...")
    drug_graphs_dict = torch.load(f'data/{dataset}/drug_graph.pt')
    target_graphs_dict = torch.load(f'data/{dataset}/target_graph.pt')
    drug_seq_embedding = np.load(f'data/{dataset}/drugs_embedding.npy')
    target_seq_embedding = np.load(f'data/{dataset}/targets_embedding.npy')

    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug", seq=drug_seq_embedding)
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=len(drug_graphs_dict))

    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target", seq=target_seq_embedding)
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=len(target_graphs_dict))

    architecture = Architecture()
    architecture.to(device)


    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)


    if fold != -100:
        best_result = [1000]

    print("start training ...")

    for epoch in range(NUM_EPOCHS):
        train(architecture, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch + 1, TRAIN_BATCH_SIZE)
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader,target_graphs_DataLoader)
        result = model_evaluate(G, P, dataset)
        print(result)

        if fold != -100 and result[0] < best_result[0]:
            best_result = result

            checkpoint_path = f"models/architecture/{dataset}/benchmark/cross_validation/{model_name}.pt"
            torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

            checkpoint_path = f"models/predictor/{dataset}/benchmark/cross_validation/{model_name}.pt"
            torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

    if fold == -100:
        checkpoint_path = f"models/architecture/{dataset}/benchmark/test/{model_name}.pt"
        torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        checkpoint_path = f"models/predictor/{dataset}/benchmark/test/{model_name}.pt"
        torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        print('\npredicting for test data')

        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("reslut:", best_result)


if __name__ == '__main__':
    seed = 1
    setup_seed(seed)
    train_test()
