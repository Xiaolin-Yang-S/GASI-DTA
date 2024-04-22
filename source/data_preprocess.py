import json
from collections import OrderedDict
import torch
from GraphInput import getDrugMolecularGraph, getTargetMolecularGraph

from transformers import AutoModelWithLMHead, AutoTokenizer, T5Tokenizer, T5EncoderModel
import re
import numpy as np


def saveDrugGraph(dataset):

    drugs = json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict)
    drug_graph_dict = getDrugMolecularGraph(drugs)
    torch.save(drug_graph_dict, f'data/{dataset}/drug_graph.pt')

def getDrugGraph():
    saveDrugGraph('davis')
    saveDrugGraph('kiba')


def saveTargetGraph(dataset):

    targets = json.load(open(f'data/{dataset}/targets.txt'), object_pairs_hook=OrderedDict)
    target_graph_dict = getTargetMolecularGraph(targets, dataset)
    torch.save(target_graph_dict,f'data/{dataset}/drug_graph.pt')

def getTargetGraph():
    saveTargetGraph('davis')
    saveTargetGraph('kiba')


def saveDrugFeature(dataset):

    drugs = json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict)

    # When the sequence length is greater than 512, the pre-trained model cannot handle it
    sequences = []
    long_index = []
    index = 0
    for key, value in drugs.items():

        if len(value) > 512:
            sequences.append(value[:484])
            sequences.append(value[484:])
            long_index.append(index)
            index += 2
        else:
            sequences.append(value)
            index += 1

    drug_model = AutoModelWithLMHead.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to('cuda:0')
    drug_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")


    atoms_nums = []
    for i in range(len(sequences)):
        seq = sequences[i]
        ids = drug_tokenizer(seq, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to('cuda:0')

        atoms_nums.append(input_ids.shape[0] - 2)

    mid_embedding = []
    drug_embeddings = []
    for i in range(int(len(sequences) / 2)):
        seq = sequences[i * 2:i * 2 + 2]
        if i == int(len(sequences) / 2) and len(sequences) % 2 != 0:
            seq = sequences[i * 2:i * 2 + 3]

        ids = drug_tokenizer(seq, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to('cuda:0')
        attention_mask = torch.tensor(ids['attention_mask']).to('cuda:0')

        with torch.no_grad():
            embedding_repr = drug_model.roberta(input_ids=input_ids, attention_mask=attention_mask)

        for j in range(embedding_repr.last_hidden_state.shape[0]):

            drug_size = atoms_nums[i * 2 + j]
            emb = embedding_repr.last_hidden_state[j][1:drug_size + 1]

            if i * 2 + j in long_index:
                mid_embedding = emb
            elif i * 2 + j - 1 in long_index:
                mid_embedding = torch.concat((mid_embedding, emb), 0)
                drug_embeddings.append(np.array(mid_embedding.mean(dim=0).to('cpu')))
            else:
                drug_embeddings.append(np.array(emb.mean(dim=0).to('cpu')))

    drug_embeddings = np.array(drug_embeddings)
    print('sequence feature acquisition of {0} dataset drugs is complete, The feature dimension is {1}'.format(dataset,drug_embeddings.shape))
    np.save(f'data/{dataset}/drugs_embedding.npy', drug_embeddings)


def getDrugSequenceFeature():
    saveDrugFeature('davis')
    saveDrugFeature('kiba')


def saveTargetFeature(dataset):

    targets = json.load(open(f'data/{dataset}/targets.txt'), object_pairs_hook=OrderedDict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sequences_base = []

    for key, value in targets.items():
        sequences_base.append(value)


    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_base_mt_uniref50").to(device)
    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    model.full() if device == 'cpu' else model.half()

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences_base]

    target_embs = []
    for i in range(int(len(sequences) / 2)):

        seq = sequences[i * 2: i * 2 + 2]
        if i == int(len(sequences) / 2) and len(sequences) % 2 != 0:
            seq = sequences[i * 2:i * 2 + 3]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = tokenizer(seq, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        print(embedding_repr.last_hidden_state.shape)

        for j in range(embedding_repr.last_hidden_state.shape[0]):
            drug_size = len(sequences_base[i * 2 + j])
            emb = embedding_repr.last_hidden_state[j][0:drug_size]

            target_embs.append(np.array(emb.mean(dim=0).to('cpu')))

    target_embs = np.array(target_embs)
    print('sequence feature acquisition of {0} dataset targets is complete, The feature dimension is {1}'.format(dataset,target_embs.shape))
    np.save(f'data/{dataset}/targets_embedding.npy', target_embs)

def getTargetSequenceFeature():
    saveTargetFeature('davis')
    saveTargetFeature('kiba')


if __name__ == '__main__':
    getDrugGraph()
    getTargetGraph()
    getDrugSequenceFeature()
    getTargetSequenceFeature()