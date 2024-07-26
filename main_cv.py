import os
import logging
import torch
from torch.utils.data import DataLoader
from model.survival_dataset_medianfill import SurvivalDataset  # Adjust import as needed

from model.train_pathology import train_path, test_path
from model.train_genexpress import train_genex, test_genex
from model.train_dnameth import train_dnameth, test_dnameth
from model.train_mirna import train_mirna, test_mirna
from model.train_copy_num import train_copynum, test_copynum
from model.train_protein import train_protein, test_protein
from model.train_clinical import train, test

from model.test_ensemble import test_ensemble, test_ensemble_gridsearch
from model.options import parse_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from model.finetune_uni import finetune_uni, get_uni_embeddings, save_uni, save_embedding
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
        train_data = train_data_luad
        test_data = test_data_luad
         
  
        # train unimodal models
        
        model1, optimizer, metric_logger = train(opt, train_data, test_data, device)
        _, cindex_one,embeddings_one, case_ids_one = test(opt, model1, train_data, device) #, embeddings_one, case_ids_one
        save_embedding(embeddings_one, case_ids_one,data_dir_luad,'modality_one.npy')
        save_embedding(embeddings_one, case_ids_one,data_dir_lusc,'modality_one.npy')
        print(f"[Final] Training set C-Index (modality 1): {cindex_one:.10f}")
        _, cindex_one,embeddings_one, case_ids_one = test(opt, model1, test_data, device)  #, embeddings_one, case_ids_one
        save_embedding(embeddings_one, case_ids_one,data_dir_luad,'modality_one.npy')
        save_embedding(embeddings_one, case_ids_one,data_dir_lusc,'modality_one.npy')
        print(f"[Final] Testing set C-Index (modality 1): {cindex_one:.10f}")
        
        model2, optimizer, metric_logger = train_path(opt, train_data, test_data, device)
        _, cindex_two, embeddings_two, case_ids_two = test_path(opt, model2, train_data, device) #, embeddings_one, case_ids_one
        save_embedding(embeddings_two, case_ids_two,data_dir_luad,'modality_two.npy')
        save_embedding(embeddings_two, case_ids_two,data_dir_lusc,'modality_two.npy')
        print(f"[Final] Training set C-Index (modality 2): {cindex_two:.10f}")
        _, cindex_two, embeddings_two, case_ids_two = test_path(opt, model2, test_data, device)  #, embeddings_one, case_ids_one
        save_embedding(embeddings_two, case_ids_two,data_dir_luad,'modality_two.npy')
        save_embedding(embeddings_two, case_ids_two,data_dir_lusc,'modality_two.npy')
        print(f"[Final] Testing set C-Index (modality 2): {cindex_two:.10f}")
        
        model3, optimizer, metric_logger = train_genex(opt, train_data, test_data, device)
        _, cindex_three, embeddings_three, case_ids_three = test_genex(opt, model3, train_data, device) #, embeddings_one, case_ids_one
        save_embedding(embeddings_three, case_ids_three,data_dir_luad,'modality_three.npy')
        save_embedding(embeddings_three, case_ids_three,data_dir_lusc,'modality_three.npy')
        print(f"[Final] Training set C-Index (modality 3): {cindex_three:.10f}")
        _, cindex_three, embeddings_three, case_ids_three = test_genex(opt, model3, test_data, device)  #, embeddings_one, case_ids_one
        save_embedding(embeddings_three, case_ids_three,data_dir_luad,'modality_three.npy')
        save_embedding(embeddings_three, case_ids_three,data_dir_lusc,'modality_three.npy')
        print(f"[Final] Testing set C-Index (modality 3): {cindex_three:.10f}")
        
        model4, optimizer, metric_logger = train_dnameth(opt, train_data, test_data, device)
        _, cindex_four, embeddings_four, case_ids_four = test_dnameth(opt, model4, train_data, device) #, embeddings_one, case_ids_one
        save_embedding(embeddings_four, case_ids_four,data_dir_luad,'modality_four.npy')
        save_embedding(embeddings_four, case_ids_four,data_dir_lusc,'modality_four.npy')
        print(f"[Final] Training set C-Index (modality 4): {cindex_four:.10f}")
        _, cindex_four, embeddings_four, case_ids_four = test_dnameth(opt, model4, test_data, device)  #, embeddings_one, case_ids_one
        save_embedding(embeddings_four, case_ids_four,data_dir_luad,'modality_four.npy')
        save_embedding(embeddings_four, case_ids_four,data_dir_lusc,'modality_four.npy')
        print(f"[Final] Testing set C-Index (modality 4): {cindex_four:.10f}")
        
        model5, optimizer, metric_logger = train_mirna(opt, train_data, test_data, device)
        _, cindex_five, embeddings_five, case_ids_five = test_mirna(opt, model5, train_data, device) #, embeddings_one, case_ids_one
        save_embedding(embeddings_five, case_ids_five,data_dir_luad,'modality_five.npy')
        save_embedding(embeddings_five, case_ids_five,data_dir_lusc,'modality_five.npy')
        print(f"[Final] Training set C-Index (modality 5): {cindex_five:.10f}")
        _, cindex_five, embeddings_five, case_ids_five = test_mirna(opt, model5, test_data, device)  #, embeddings_one, case_ids_one
        save_embedding(embeddings_five, case_ids_five,data_dir_luad,'modality_five.npy')
        save_embedding(embeddings_five, case_ids_five,data_dir_lusc,'modality_five.npy')
        print(f"[Final] Testing set C-Index (modality 5): {cindex_five:.10f}")
        
        #model6, optimizer, metric_logger = train_copynum(opt, train_data, test_data, device)
        #_, cindex_six, embeddings_six, case_ids_six = test_copynum(opt, model6, train_data, device) #, embeddings_one, case_ids_one
        #save_embedding(embeddings_six, case_ids_six,data_dir_luad,'modality_six.npy')
        #save_embedding(embeddings_six, case_ids_six,data_dir_lusc,'modality_six.npy')
        #print(f"[Final] Training set C-Index (modality 6): {cindex_six:.10f}")
        #_, cindex_six, embeddings_six, case_ids_six = test_copynum(opt, model6, test_data, device)  #, embeddings_one, case_ids_one
        #save_embedding(embeddings_six, case_ids_six,data_dir_luad,'modality_six.npy')
        #save_embedding(embeddings_six, case_ids_six,data_dir_lusc,'modality_six.npy')
        #print(f"[Final] Testing set C-Index (modality 6): {cindex_six:.10f}")
        
        model7, optimizer, metric_logger = train_protein(opt, train_data, test_data, device)
        _, cindex_seven, embeddings_seven, case_ids_seven = test_protein(opt, model7, train_data, device) #, embeddings_one, case_ids_one
        save_embedding(embeddings_seven, case_ids_seven,data_dir_luad,'modality_seven.npy')
        save_embedding(embeddings_seven, case_ids_seven,data_dir_lusc,'modality_seven.npy')
        print(f"[Final] Training set C-Index (modality 7): {cindex_seven:.10f}")
        _, cindex_seven, embeddings_seven, case_ids_seven = test_protein(opt, model7, test_data, device)  #, embeddings_one, case_ids_one
        save_embedding(embeddings_seven, case_ids_seven,data_dir_luad,'modality_seven.npy')
        save_embedding(embeddings_seven, case_ids_seven,data_dir_lusc,'modality_seven.npy')
        print(f"[Final] Testing set C-Index (modality 7): {cindex_seven:.10f}")
        
        
        print(f"All Testing C-Indexes:1. {cindex_one:.10f} ,2. {cindex_two:.10f},3. {cindex_three:.10f},4. {cindex_four:.10f},5. {cindex_five:.10f},6. {cindex_seven:.10f}")
        
        #refresh dataset with extracted embeddings(predictions)
        train_data_luad = SurvivalDataset(data_dir_luad, fold, seed,5, split='train')
        test_data_luad = SurvivalDataset(data_dir_luad, fold, seed,5, split='test')
        train_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed,5, split='train')
        test_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed,5, split='test')
        train_data = train_data_luad
        test_data = test_data_luad
        
        # train multimodal late fusion model:
        #model_late_fuse, _, _ = train_2mod(opt, train_data, test_data, device)
        #_, cindex_late = test_2mod(opt, model_late_fuse, test_data, device)
        #print(f"[Final] Testing set C-Index (late fusion): {cindex_late:.10f}")
        
        #cindex_ensemble = test_ensemble(train_data, test_data, device) 
        cindex_ensemble, best_weights = test_ensemble_gridsearch(train_data, test_data, device)
        print(f"[Final] Testing set C-Index (ensemble): {cindex_ensemble:.10f}")

        c_indices_per_fold.append(cindex_ensemble) 
        cindex_ensembles.append(cindex_ensemble)
        cindex_latefuse.append(cindex_ensemble)
        cindex_ones.append(cindex_one)
        cindex_twos.append(cindex_two)
        cindex_threes.append(cindex_three)
        cindex_fours.append(cindex_four)
        cindex_fives.append(cindex_five)
        #cindex_sixes.append(cindex_six)
        cindex_sevens.append(cindex_seven)
        best_weights_all.append(best_weights)

    average_c_index = sum(c_indices_per_fold) / len(c_indices_per_fold)
    print("Average C-Index(seed {seed}):", average_c_index)
    c_indices_per_seed.append(c_indices_per_fold)
average_c_index_final = sum(cindex_ensembles) / len(cindex_ensembles)
average_c_index_final_latefuse = sum(cindex_latefuse) / len(cindex_latefuse)
#std_dev_c_index_final = statistics.stdev(c_indices_per_seed)
# Output results

print('mod1 c-indeces:')
print(cindex_ones)
print('mod2 c-indeces:')
print(cindex_twos)
print('mod3 c-indeces:')
print(cindex_threes)
print('mod4 c-indeces:')
print(cindex_fours)
print('mod5 c-indeces:')
print(cindex_fives)
#print('mod6 c-indeces:')
#print(cindex_sixes)
print('mod7 c-indeces:')
print(cindex_sevens)

print('ensemble c-indeces')
print(c_indices_per_seed)
print(f"[Final] Average C-Index across all 5-fold tests: {average_c_index_final:.10f}")
print('grid search weights:')
print(best_weights_all)
#print(f"[Final] Standard deviation across all 5-fold tests: {std_dev_c_index_final:.10f}")

print('late fuse c-indices')
print(cindex_latefuse)
print('average of these:')
print(average_c_index_final_latefuse)
