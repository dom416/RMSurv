import os
import logging
import torch
from torch.utils.data import DataLoader
from model.survival_dataset_medianfill import SurvivalDataset  # Adjust import as needed

from model.train_concat import train_concat, test_concat
from model.train_concat_cox import train_concat_cox, test_concat_cox
from model.train_HFB_6 import train_HFB, test_HFB
from model.train_HFB_cox_6 import train_HFB_cox, test_HFB_cox




from model.train_test_2modal import train_2mod, test_2mod
from model.options import parse_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from model.finetune_uni import finetune_uni, get_uni_embeddings, save_uni, save_embedding
from SeNMomain.SeNMomain.SENMO import train_senmo
import statistics

# Initialize parser and device
opt = parse_args()
device = torch.device('cuda:2')
print("Using device:", device)
data_dir_luad = '/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/LUAD/'
data_dir_lusc = '/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/LUSC/'
c_indices_per_seed = []
cindex_ones, cindex_twos, cindex_threes, cindex_fours, cindex_fives, cindex_sixes,cindex_sevens,cindex_ensembles, cindex_latefuse,best_weights_all = [],[],[],[],[],[],[],[],[],[]

for seed in range(1,11):

    c_indices_per_fold = []
    for fold in range(1, 6):
        print('seed:')
        print(seed)
        print('fold:')
        print(fold)

        train_data_luad = SurvivalDataset(data_dir_luad, fold, seed,5, split='train')
        test_data_luad = SurvivalDataset(data_dir_luad, fold, seed,5, split='test')
        train_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed,5, split='train')
        test_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed,5, split='test')
        train_data = train_data_luad + train_data_lusc
        test_data = test_data_luad + test_data_lusc
         
  
        # train unimodal models
        
        #model1, optimizer, metric_logger = train_concat(opt, train_data, test_data, device)
        #_, cindex_one,embeddings_one, case_ids_one = test_concat(opt, model1, train_data, device) #, embeddings_one, case_ids_one
        #print(f"[Final] Training set C-Index (modality 1): {cindex_one:.10f}")
        #_, cindex_one,embeddings_one, case_ids_one = test_concat(opt, model1, test_data, device)  #, embeddings_one, case_ids_one
        #print(f"[Final] Testing set C-Index (modality 1): {cindex_one:.10f}")
        
        model1, optimizer, metric_logger = train_concat_cox(opt, train_data, test_data, device)
        _, cindex_one = test_concat_cox(opt, model1, train_data, device) #, embeddings_one, case_ids_one
        print(f"[Final] Training set C-Index (modality 1): {cindex_one:.10f}")
        _, cindex_one = test_concat_cox(opt, model1, test_data, device)  #, embeddings_one, case_ids_one
        print(f"[Final] Testing set C-Index (modality 1): {cindex_one:.10f}")
        
        #model1, optimizer, metric_logger = train_HFB(opt, train_data, test_data, device)
        #_, cindex_one, embeddings_one, case_ids_one = test_HFB(opt, model1, train_data, device) #, embeddings_one, case_ids_one
        #print(f"[Final] Training set C-Index (modality 1): {cindex_one:.10f}")
        #_, cindex_one, embeddings_one, case_ids_one = test_HFB(opt, model1, test_data, device)  #, embeddings_one, case_ids_one
        #print(f"[Final] Testing set C-Index (modality 1): {cindex_one:.10f}")
        
        #model2, optimizer, metric_logger = train_HFB_cox(opt, train_data, test_data, device)
        #_, cindex_two = test_HFB_cox(opt, model2, train_data, device) #, embeddings_one, case_ids_one
        #print(f"[Final] Training set C-Index (modality 2): {cindex_two:.10f}")
        #_, cindex_two = test_HFB_cox(opt, model2, test_data, device)  #, embeddings_one, case_ids_one
        #print(f"[Final] Testing set C-Index (modality 2): {cindex_two:.10f}")
        

        cindex_ones.append(cindex_one)
        #cindex_twos.append(cindex_two)
#std_dev_c_index_final = statistics.stdev(c_indices_per_seed)
# Output results

print('mod1 c-indeces:')
print(cindex_ones)
#print('mod2 c-indeces:')
#print(cindex_twos)
