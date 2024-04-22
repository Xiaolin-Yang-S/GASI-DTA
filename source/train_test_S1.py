import os, json, torch
import numpy as np
from collections import OrderedDict


from model import GASIDTA_cold_drug, Predictor
from metrics import model_evaluate
from GraphInput import getDrugMolecularGraph, getTargetMolecularGraph
from utils import argparser, DTADataset, GraphDataset, collate, predicting, read_data, train, setup_seed

import warnings
warnings.filterwarnings('ignore')



def create_dataset_for_train_test(affinity, dataset, fold):

    dataset_path = 'data/' + dataset + '/'

    drug_train_fold_origin = json.load(open(dataset_path + 'S1_train_set.txt'))
    drug_train_folds = []
    for i in range(len(drug_train_fold_origin)):
        if i != fold:
            drug_train_folds += drug_train_fold_origin[i]
    drug_test_fold = json.load(open(dataset_path + 'S1_test_set.txt')) if fold == -100 else drug_train_fold_origin[fold]

    # train set and test set
    train_affinity = affinity[drug_train_folds, :]
    test_affinity = affinity[drug_test_fold, :]

    train_rows, train_cols = np.where(np.isnan(train_affinity) == False)
    train_Y = train_affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

    test_rows, test_cols = np.where(np.isnan(test_affinity) == False)
    test_Y = test_affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity[np.isnan(train_affinity) == True] = 0

    # drug molecular graphs
    drugs = json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict)

    drugs_seq = np.load(f'data/{dataset}/drugs_embedding.npy')

    drug_keys = np.array(list(drugs.keys()))
    drug_values = np.array(list(drugs.values()))
    train_drug_keys = drug_keys[drug_train_folds]
    train_drug_values = drug_values[drug_train_folds]

    train_drug_seq = drugs_seq[drug_train_folds]

    train_drugs = dict(zip(train_drug_keys, train_drug_values))
    test_drug_keys = drug_keys[drug_test_fold]
    test_drug_values = drug_values[drug_test_fold]

    test_drug_seq = drugs_seq[drug_test_fold]

    test_drugs = dict(zip(test_drug_keys, test_drug_values))

    train_drug_graphs_dict = getDrugMolecularGraph(train_drugs)
    test_drug_graphs_dict = getDrugMolecularGraph(test_drugs)

    target_graphs_dict = torch.load(f'data/{dataset}/target_graph.pt')

    return (train_dataset, test_dataset, train_drug_graphs_dict, test_drug_graphs_dict, target_graphs_dict, train_drug_seq, test_drug_seq)


def train_test():

    FLAGS = argparser()

    setup_seed(FLAGS.seed)

    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    NUM_EPOCHS = FLAGS.num_epochs
    LR = FLAGS.lr
    Architecture = GASIDTA_cold_drug
    model_name = Architecture.__name__
    fold = FLAGS.fold
    if fold != -100:
        model_name += f"-{FLAGS.drug_sim_k}-{fold}"

    NUM_EPOCHS = 300
    cuda_name = f'cuda:1'
    dataset = 'davis'

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("seed:", FLAGS.seed)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LR)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)
    
    if os.path.exists(f"models/architecture/{dataset}/S2/cross_validation/") is False:
        os.makedirs(f"models/architecture/{dataset}/S2/cross_validation/")
    if os.path.exists(f"models/predictor/{dataset}/S2/cross_validation/") is False:
        os.makedirs(f"models/predictor/{dataset}/S2/cross_validation/")
    if os.path.exists(f"models/architecture/{dataset}/S2/test/") is False:
        os.makedirs(f"models/architecture/{dataset}/S2/test/")
    if os.path.exists(f"models/predictor/{dataset}/S2/test/") is False:
        os.makedirs(f"models/predictor/{dataset}/S2/test/")

    print("create dataset ...")
    affinity = read_data(dataset)
    (train_data, test_data, train_drug_graphs_dict, test_drug_graphs_dict, target_graphs_dict,
      train_drug_seq, test_drug_seq) = create_dataset_for_train_test(affinity, dataset, fold)

    print("create train_loader and test_loader ...")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    targets_seq = np.load(f'data/{dataset}/targets_embedding.npy')

    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    train_drug_graphs_Data = GraphDataset(graphs_dict=train_drug_graphs_dict, dttype="drug", seq=train_drug_seq)
    train_drug_graphs_DataLoader = torch.utils.data.DataLoader(train_drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=len(train_drug_graphs_dict))
    test_drug_graphs_Data = GraphDataset(graphs_dict=test_drug_graphs_dict, dttype="drug", seq=test_drug_seq)
    test_drug_graphs_DataLoader = torch.utils.data.DataLoader(test_drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=len(test_drug_graphs_dict))
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target", seq=targets_seq)
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate, batch_size=len(target_graphs_dict))

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    architecture = Architecture()
    architecture.to(device)

    print(architecture)

    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)
    

    if fold != -100:
        best_result = [1000]
    print("start training ...")

    for epoch in range(NUM_EPOCHS):
        train(architecture, predictor, device, train_loader, train_drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch + 1, TRAIN_BATCH_SIZE)
        G, P = predicting(architecture, predictor, device, test_loader, test_drug_graphs_DataLoader, target_graphs_DataLoader)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
        if fold != -100 and result[0] < best_result[0]:
            best_result = result

            checkpoint_path = f"models/architecture/{dataset}/S1/cross_validation/{model_name}.pt"
            torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

            checkpoint_path = f"models/predictor/{dataset}/S1/cross_validation/{model_name}.pt"
            torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

    if fold == -100:
        checkpoint_path = f"models/architecture/{dataset}/S1/test/{model_name}.pt"
        torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        checkpoint_path = f"models/predictor/{dataset}/S1/test/{model_name}.pt"
        torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        print('\npredicting for test data')
        G, P = predicting(architecture, predictor, device, test_loader, test_drug_graphs_DataLoader, target_graphs_DataLoader)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("reslut:", best_result)


if __name__ == '__main__':
    train_test()
