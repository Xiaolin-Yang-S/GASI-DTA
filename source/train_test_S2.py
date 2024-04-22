import os, json, torch
import numpy as np
from collections import OrderedDict



from model import GASIDTA_cold_target, Predictor
from metrics import model_evaluate
from GraphInput import getTargetMolecularGraph
from utils import argparser, DTADataset, GraphDataset, collate, predicting, read_data, train, setup_seed

import warnings
warnings.filterwarnings('ignore')



def create_dataset_for_train_test(affinity, dataset, fold):
    # load dataset
    dataset_path = 'data/' + dataset + '/'

    target_train_fold_origin = json.load(open(dataset_path + 'S1_train_set.txt'))
    target_train_folds = []
    for i in range(len(target_train_fold_origin)):
        if i != fold:
            target_train_folds += target_train_fold_origin[i]
    target_test_fold = json.load(open(dataset_path + 'S1_test_set.txt')) if fold == -100 else target_train_fold_origin[fold]

    # train set and test set
    train_affinity = affinity[:, target_train_folds]
    test_affinity = affinity[:, target_test_fold]

    train_rows, train_cols = np.where(np.isnan(train_affinity) == False)
    train_Y = train_affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

    test_rows, test_cols = np.where(np.isnan(test_affinity) == False)
    test_Y = test_affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity[np.isnan(train_affinity) == True] = 0

    # target molecular graphs
    targets = json.load(open(f'data/{dataset}/targets.txt'), object_pairs_hook=OrderedDict)

    targets_seq = np.load(f'data/{dataset}/targets_embedding.npy')

    target_keys = np.array(list(targets.keys()))
    target_values = np.array(list(targets.values()))

    train_targets_seq = targets_seq[target_train_folds]

    train_target_keys = target_keys[target_train_folds]
    train_target_values = target_values[target_train_folds]
    train_targets = dict(zip(train_target_keys, train_target_values))

    test_targets_seq = targets_seq[target_test_fold]

    test_target_keys = target_keys[target_test_fold]
    test_target_values = target_values[target_test_fold]
    test_targets = dict(zip(test_target_keys, test_target_values))

    train_target_graphs_dict = getTargetMolecularGraph(train_targets, dataset)
    test_target_graphs_dict = getTargetMolecularGraph(test_targets, dataset)

    drug_graphs_dict = torch.load(f'data/{dataset}/drug_graph.pt')

    return (train_dataset, test_dataset, drug_graphs_dict,
            train_target_graphs_dict, test_target_graphs_dict,
            train_targets_seq, test_targets_seq)


def train_test():

    FLAGS = argparser()

    setup_seed(FLAGS.seed)

    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    NUM_EPOCHS = FLAGS.num_epochs
    LR = FLAGS.lr
    Architecture = GASIDTA_cold_target
    model_name = Architecture.__name__
    fold = FLAGS.fold
    if fold != -100:
        model_name += f"-{FLAGS.target_sim_k}-{fold}"

    NUM_EPOCHS = 300
    dataset = 'kiba'
    cuda_name = f'cuda:1'

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("seed:", FLAGS.seed)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LR)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)
    
    if os.path.exists(f"models/architecture/{dataset}/S3/cross_validation/") is False:
        os.makedirs(f"models/architecture/{dataset}/S3/cross_validation/")
    if os.path.exists(f"models/predictor/{dataset}/S3/cross_validation/") is False:
        os.makedirs(f"models/predictor/{dataset}/S3/cross_validation/")
    if os.path.exists(f"models/architecture/{dataset}/S3/test/") is False:
        os.makedirs(f"models/architecture/{dataset}/S3/test/")
    if os.path.exists(f"models/predictor/{dataset}/S3/test/") is False:
        os.makedirs(f"models/predictor/{dataset}/S3/test/")

    print("create dataset ...")
    affinity = read_data(dataset)
    (train_data, test_data, drug_graphs_dict,
     train_target_graphs_dict, test_target_graphs_dict, train_targets_seq, test_targets_seq) = create_dataset_for_train_test(affinity, dataset, fold)
    print("create train_loader and test_loader ...")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    drugs_seq = np.load(f'data/{dataset}/drugs_embedding.npy')
    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug", seq = drugs_seq)
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=len(drug_graphs_dict))
    train_target_graphs_Data = GraphDataset(graphs_dict=train_target_graphs_dict, dttype="target", seq = train_targets_seq)
    train_target_graphs_DataLoader = torch.utils.data.DataLoader(train_target_graphs_Data, shuffle=False, collate_fn=collate, batch_size=len(train_target_graphs_dict))
    test_target_graphs_Data = GraphDataset(graphs_dict=test_target_graphs_dict, dttype="target", seq = test_targets_seq)
    test_target_graphs_DataLoader = torch.utils.data.DataLoader(test_target_graphs_Data, shuffle=False, collate_fn=collate, batch_size=len(test_target_graphs_dict))

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    architecture = Architecture()
    architecture.to(device)

    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)


    if fold != -100:
        best_result = [1000]

    print("start training ...")
    for epoch in range(NUM_EPOCHS):
        train(architecture, predictor, device, train_loader, drug_graphs_DataLoader, train_target_graphs_DataLoader, LR, epoch + 1, TRAIN_BATCH_SIZE)
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader, test_target_graphs_DataLoader)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
        if fold != -100 and result[0] < best_result[0]:
            best_result = result
            checkpoint_dir = f"models/architecture/{dataset}/S3/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)
            
            checkpoint_dir = f"models/predictor/{dataset}/S3/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

    if fold == -100:
        checkpoint_dir = f"models/architecture/{dataset}/S3/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        checkpoint_dir = f"models/predictor/{dataset}/S3/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        print('\npredicting for test data')
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader, test_target_graphs_DataLoader)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("reslut:", best_result)



if __name__ == '__main__':
    train_test()
