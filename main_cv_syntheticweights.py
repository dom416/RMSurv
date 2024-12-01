import os
import logging
import torch
from torch.utils.data import DataLoader
from HFBSurvmain.HFBSurvmain.HFBSurv.model.survival_dataset_medianfill import SurvivalDataset  # Adjust import as needed

from model.train_pathology import train_path, test_path
from model.train_genexpress import train_genex, test_genex
from model.train_dnameth import train_dnameth, test_dnameth
from model.train_mirna import train_mirna, test_mirna 
from model.train_copy_num import train_copynum, test_copynum
from model.train_protein import train_protein, test_protein
from model.train_clinical import train, test

from HFBSurvmain.HFBSurvmain.HFBSurv.model.train_test_2modal import train_2mod, test_2mod

from HFBSurvmain.HFBSurvmain.HFBSurv.model.test_ensemble_1to6mods import test_ensemble, test_ensemble_weight_finding, compute_rho_between

from HFBSurvmain.HFBSurvmain.HFBSurv.model.test_ensemble import test_ensemble_stdev, test_ensemble_weight_finding_stdev


from HFBSurvmain.HFBSurvmain.HFBSurv.model.options import parse_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HFBSurvmain.HFBSurvmain.HFBSurv.model.finetune_uni import finetune_uni, get_uni_embeddings, save_uni, save_embedding
from SeNMomain.SeNMomain.SENMO import train_senmo
import statistics
from compute_ideal_weights_2 import compute_ideal_weights
from compute_ideal_weights_stdev import compute_ideal_weights_stdev

# Initialize parser and device
opt = parse_args()
device = torch.device('cuda:2')
print("Using device:", device)
data_dir_luad = '/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/LUAD/'
data_dir_lusc = '/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/LUAD_LUSC_data/LUSC/'

# Prepare to store results
c_indices_per_seed = [] 
cindex_ensembles,cindex_ensembles_synth,cindex_ensembles_adhoc,cindex_ensembles_stdev,cindex_ensembles_stdev_synth, best_weights_all = [],[],[],[],[],[]
c_indices_synth = {N: [] for N in range(2, 7)}  # N from 2 to 6
c_indices_synth_stdev = {N: [] for N in range(2, 7)}
c_indices_adhoc = {N: [] for N in range(2, 7)}

# Outer Loop: 10 Seeds
for seed in range(1, 11): 
    
    c_indices_per_fold = []
    best_weights_fold = []

    # Outer CV Loop: 5 CV Folds
    for fold in range(1, 6):
    
        print(f"Seed: {seed}, Fold: {fold}")

        # Inner Nested CV: 5 Validation Folds for Grid Search
        best_weights_nested = []
        best_weights_nested_stdev = []
        
        c_i_1s_nested,c_i_2s_nested,c_i_3s_nested,c_i_4s_nested,c_i_5s_nested,c_i_6s_nested = [],[],[],[],[],[]

        for val_fold in range(1, 6):
          
            # Load the datasets
            train_data_luad = SurvivalDataset(data_dir_luad, fold, seed, 5, split='train', nested_cv=True, cv_fold=val_fold)
            test_data_luad = SurvivalDataset(data_dir_luad, fold, seed, 5, split='val', nested_cv=True, cv_fold=val_fold)
            train_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed, 5, split='train', nested_cv=True, cv_fold=val_fold)
            test_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed, 5, split='val', nested_cv=True, cv_fold=val_fold)
            train_data = train_data_luad
            test_data = test_data_luad

            # Train unimodal models, test, and save embeddings for each modality

            model1, optimizer, metric_logger = train(opt, train_data, test_data, device)
            _, cindex_one, embeddings_one, case_ids_one = test(opt, model1, train_data, device)
            save_embedding(embeddings_one, case_ids_one, data_dir_luad, 'modality_one.npy')
            save_embedding(embeddings_one, case_ids_one, data_dir_lusc, 'modality_one.npy')
            _, cindex_one, embeddings_one, case_ids_one = test(opt, model1, test_data, device)
            save_embedding(embeddings_one, case_ids_one, data_dir_luad, 'modality_one.npy')
            save_embedding(embeddings_one, case_ids_one, data_dir_lusc, 'modality_one.npy')

            model2, optimizer, metric_logger = train_path(opt, train_data, test_data, device)
            _, cindex_two, embeddings_two, case_ids_two = test_path(opt, model2, train_data, device)
            save_embedding(embeddings_two, case_ids_two, data_dir_luad, 'modality_two.npy')
            save_embedding(embeddings_two, case_ids_two, data_dir_lusc, 'modality_two.npy')
            _, cindex_two, embeddings_two, case_ids_two = test_path(opt, model2, test_data, device)
            save_embedding(embeddings_two, case_ids_two, data_dir_luad, 'modality_two.npy')
            save_embedding(embeddings_two, case_ids_two, data_dir_lusc, 'modality_two.npy')

            model3, optimizer, metric_logger = train_genex(opt, train_data, test_data, device)
            _, cindex_three, embeddings_three, case_ids_three = test_genex(opt, model3, train_data, device)
            save_embedding(embeddings_three, case_ids_three, data_dir_luad, 'modality_three.npy')
            save_embedding(embeddings_three, case_ids_three, data_dir_lusc, 'modality_three.npy')
            _, cindex_three, embeddings_three, case_ids_three = test_genex(opt, model3, test_data, device)
            save_embedding(embeddings_three, case_ids_three, data_dir_luad, 'modality_three.npy')
            save_embedding(embeddings_three, case_ids_three, data_dir_lusc, 'modality_three.npy')

            model4, optimizer, metric_logger = train_dnameth(opt, train_data, test_data, device)
            _, cindex_four, embeddings_four, case_ids_four = test_dnameth(opt, model4, train_data, device)
            save_embedding(embeddings_four, case_ids_four, data_dir_luad, 'modality_four.npy')
            save_embedding(embeddings_four, case_ids_four, data_dir_lusc, 'modality_four.npy')
            _, cindex_four, embeddings_four, case_ids_four = test_dnameth(opt, model4, test_data, device)
            save_embedding(embeddings_four, case_ids_four, data_dir_luad, 'modality_four.npy')
            save_embedding(embeddings_four, case_ids_four, data_dir_lusc, 'modality_four.npy')

            model5, optimizer, metric_logger = train_mirna(opt, train_data, test_data, device)
            _, cindex_five, embeddings_five, case_ids_five = test_mirna(opt, model5, train_data, device)
            save_embedding(embeddings_five, case_ids_five, data_dir_luad, 'modality_five.npy')
            save_embedding(embeddings_five, case_ids_five, data_dir_lusc, 'modality_five.npy')
            _, cindex_five, embeddings_five, case_ids_five = test_mirna(opt, model5, test_data, device)
            save_embedding(embeddings_five, case_ids_five, data_dir_luad, 'modality_five.npy')
            save_embedding(embeddings_five, case_ids_five, data_dir_lusc, 'modality_five.npy')

            model7, optimizer, metric_logger = train_protein(opt, train_data, test_data, device)
            _, cindex_seven, embeddings_seven, case_ids_seven = test_protein(opt, model7, train_data, device)
            save_embedding(embeddings_seven, case_ids_seven, data_dir_luad, 'modality_seven.npy')
            save_embedding(embeddings_seven, case_ids_seven, data_dir_lusc, 'modality_seven.npy')
            _, cindex_seven, embeddings_seven, case_ids_seven = test_protein(opt, model7, test_data, device)
            save_embedding(embeddings_seven, case_ids_seven, data_dir_luad, 'modality_seven.npy')
            save_embedding(embeddings_seven, case_ids_seven, data_dir_lusc, 'modality_seven.npy')
            
            # reload data
            train_data_luad = SurvivalDataset(data_dir_luad, fold, seed, 5, split='train', nested_cv=True, cv_fold=val_fold)
            test_data_luad = SurvivalDataset(data_dir_luad, fold, seed, 5, split='val', nested_cv=True, cv_fold=val_fold)
            train_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed, 5, split='train', nested_cv=True, cv_fold=val_fold)
            test_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed, 5, split='val', nested_cv=True, cv_fold=val_fold)
            train_data = train_data_luad
            test_data = test_data_luad

            c_i_1s_nested.append(cindex_one)
            c_i_2s_nested.append(cindex_two)
            c_i_3s_nested.append(cindex_three)
            c_i_4s_nested.append(cindex_four)
            c_i_5s_nested.append(cindex_five)
            c_i_6s_nested.append(cindex_seven)
        
        avg_c_i_1 = np.mean(c_i_1s_nested,axis=0)
        avg_c_i_2 = np.mean(c_i_2s_nested,axis=0)
        avg_c_i_3 = np.mean(c_i_3s_nested,axis=0)
        avg_c_i_4 = np.mean(c_i_4s_nested,axis=0)
        avg_c_i_5 = np.mean(c_i_5s_nested,axis=0)
        avg_c_i_6 = np.mean(c_i_6s_nested,axis=0)
        average_val_cindices = [avg_c_i_1,avg_c_i_2,avg_c_i_3,avg_c_i_4,avg_c_i_5,avg_c_i_6]
        
        best_weights_adhoc = [0 if avg_c_i_1 < 0.52 else avg_c_i_1 - 0.5,0 if avg_c_i_2 < 0.52 else avg_c_i_2 - 0.5,0 if avg_c_i_3 < 0.52 else avg_c_i_3 - 0.5,0 if avg_c_i_4 < 0.52 else avg_c_i_4 - 0.5,0 if avg_c_i_5 < 0.52 else avg_c_i_5 - 0.5,0 if avg_c_i_6 < 0.52 else avg_c_i_6 - 0.5]


        # Re-train models for this fold and use the saved ensemble weights
        train_data_luad = SurvivalDataset(data_dir_luad, fold, seed, 5, split='train', nested_cv=False, cv_fold=None)
        test_data_luad = SurvivalDataset(data_dir_luad, fold, seed, 5, split='test', nested_cv=False, cv_fold=None)
        train_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed, 5, split='train', nested_cv=False, cv_fold=None)
        test_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed, 5, split='test', nested_cv=False, cv_fold=None)
        train_data = train_data_luad
        test_data = test_data_luad

        # Train unimodal models, test, and save embeddings for each modality
        model1, optimizer, metric_logger = train(opt, train_data, test_data, device)
        _, cindex_one, embeddings_one, case_ids_one = test(opt, model1, train_data, device)
        save_embedding(embeddings_one, case_ids_one, data_dir_luad, 'modality_one.npy')
        save_embedding(embeddings_one, case_ids_one, data_dir_lusc, 'modality_one.npy')
        _, cindex_one, embeddings_one, case_ids_one = test(opt, model1, test_data, device)
        save_embedding(embeddings_one, case_ids_one, data_dir_luad, 'modality_one.npy')
        save_embedding(embeddings_one, case_ids_one, data_dir_lusc, 'modality_one.npy')

        model2, optimizer, metric_logger = train_path(opt, train_data, test_data, device)
        _, cindex_two, embeddings_two, case_ids_two = test_path(opt, model2, train_data, device)
        save_embedding(embeddings_two, case_ids_two, data_dir_luad, 'modality_two.npy')
        save_embedding(embeddings_two, case_ids_two, data_dir_lusc, 'modality_two.npy')
        _, cindex_two, embeddings_two, case_ids_two = test_path(opt, model2, test_data, device)
        save_embedding(embeddings_two, case_ids_two, data_dir_luad, 'modality_two.npy')
        save_embedding(embeddings_two, case_ids_two, data_dir_lusc, 'modality_two.npy')

        model3, optimizer, metric_logger = train_genex(opt, train_data, test_data, device)
        _, cindex_three, embeddings_three, case_ids_three = test_genex(opt, model3, train_data, device)
        save_embedding(embeddings_three, case_ids_three, data_dir_luad, 'modality_three.npy')
        save_embedding(embeddings_three, case_ids_three, data_dir_lusc, 'modality_three.npy')
        _, cindex_three, embeddings_three, case_ids_three = test_genex(opt, model3, test_data, device)
        save_embedding(embeddings_three, case_ids_three, data_dir_luad, 'modality_three.npy')
        save_embedding(embeddings_three, case_ids_three, data_dir_lusc, 'modality_three.npy')

        model4, optimizer, metric_logger = train_dnameth(opt, train_data, test_data, device)
        _, cindex_four, embeddings_four, case_ids_four = test_dnameth(opt, model4, train_data, device)
        save_embedding(embeddings_four, case_ids_four, data_dir_luad, 'modality_four.npy')
        save_embedding(embeddings_four, case_ids_four, data_dir_lusc, 'modality_four.npy')
        _, cindex_four, embeddings_four, case_ids_four = test_dnameth(opt, model4, test_data, device)
        save_embedding(embeddings_four, case_ids_four, data_dir_luad, 'modality_four.npy')
        save_embedding(embeddings_four, case_ids_four, data_dir_lusc, 'modality_four.npy')

        model5, optimizer, metric_logger = train_mirna(opt, train_data, test_data, device)
        _, cindex_five, embeddings_five, case_ids_five = test_mirna(opt, model5, train_data, device)
        save_embedding(embeddings_five, case_ids_five, data_dir_luad, 'modality_five.npy')
        save_embedding(embeddings_five, case_ids_five, data_dir_lusc, 'modality_five.npy')
        _, cindex_five, embeddings_five, case_ids_five = test_mirna(opt, model5, test_data, device)
        save_embedding(embeddings_five, case_ids_five, data_dir_luad, 'modality_five.npy')
        save_embedding(embeddings_five, case_ids_five, data_dir_lusc, 'modality_five.npy')

        model7, optimizer, metric_logger = train_protein(opt, train_data, test_data, device)
        _, cindex_seven, embeddings_seven, case_ids_seven = test_protein(opt, model7, train_data, device)
        save_embedding(embeddings_seven, case_ids_seven, data_dir_luad, 'modality_seven.npy')
        save_embedding(embeddings_seven, case_ids_seven, data_dir_lusc, 'modality_seven.npy')
        _, cindex_seven, embeddings_seven, case_ids_seven = test_protein(opt, model7, test_data, device)
        save_embedding(embeddings_seven, case_ids_seven, data_dir_luad, 'modality_seven.npy')
        save_embedding(embeddings_seven, case_ids_seven, data_dir_lusc, 'modality_seven.npy')
        
        
        #Reload data
        train_data_luad = SurvivalDataset(data_dir_luad, fold, seed, 5, split='train', nested_cv=False, cv_fold=None)
        test_data_luad = SurvivalDataset(data_dir_luad, fold, seed, 5, split='test', nested_cv=False, cv_fold=None)
        train_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed, 5, split='train', nested_cv=False, cv_fold=None)
        test_data_lusc = SurvivalDataset(data_dir_lusc, fold, seed, 5, split='test', nested_cv=False, cv_fold=None)
        train_data = train_data_luad
        test_data = test_data_luad
        
        # Define the full list of modalities and their names
        modalities_order_full = ['modality_one', 'modality_five', 'modality_three',
                                 'modality_four', 'modality_two', 'modality_seven']
        
        modality_names_full = ['Modality 1', 'Modality 5', 'Modality 3',
                               'Modality 4', 'Modality 2', 'Modality 6']
        
        # Mapping from your modality names to the expected names
        modality_name_mapping = {
            'modality_one': 'Modality 1',
            'modality_five': 'Modality 5',
            'modality_three': 'Modality 3',
            'modality_four': 'Modality 4',
            'modality_two': 'Modality 2',
            'modality_seven': 'Modality 6'
        }
        
        # Define the mapping from modality keys to their corresponding c-indices
        modality_cindex_mapping = {
            'modality_one': avg_c_i_1,
            'modality_two': avg_c_i_2,
            'modality_three': avg_c_i_3,
            'modality_four': avg_c_i_4,
            'modality_five': avg_c_i_5,
            'modality_seven': avg_c_i_6
        }
        
        # Create desired_cindices by ordering according to modalities_order_full
        desired_cindices_full = [modality_cindex_mapping[modality] for modality in modalities_order_full]
        
        # Calculate ad-hoc weights based on desired_cindices_full
        best_weights_adhoc_full = [0 if cindex < 0.52 else cindex - 0.5 for cindex in desired_cindices_full]
        
        for N in range(2, 7):
            # Adjust the modalities and c-indices accordingly
            modalities_order_N = modalities_order_full[:N]
            modality_names_N = modality_names_full[:N]
            desired_cindices_N = desired_cindices_full[:N]
            adhoc_weights_N = best_weights_adhoc_full[:N]
            
            # Map your modalities_order_N to the expected names
            expected_modalities_order_N = [modality_name_mapping[name] for name in modalities_order_N]
            
            # Calculate test set Pearson correlations using the updated function
            rho_between_N = compute_rho_between(
                train_data, test_data, device, modalities_order=modalities_order_N
            )
            
            # Compute synthetic weights for N modalities using expected modality names
            best_weights_synth_N, _, _ = compute_ideal_weights(
                num_modalities=N,
                desired_cindices=desired_cindices_N,
                rho_between=rho_between_N,
                N=20000,
                num_runs=15,
                num_weights=100
            )
            best_weights_synth_N = best_weights_synth_N[-1, :]
            
            # Compute synthetic weights for N modalities using expected modality names
            best_weights_synth_N_stdev, _, _, _ = compute_ideal_weights_stdev(
                num_modalities=N,
                desired_cindices=desired_cindices_N,
                rho_between=rho_between_N,
                N=20000,
                num_weights=100
            )
            
            
             # Test ensemble with synthetic weights
            cindex_ensemble_synth_N = test_ensemble(
                train_data, test_data, device, best_weights_synth_N, N, modalities_order_N
            )
        
            # Test ensemble with synthetic weights
            cindex_ensemble_synth_N_stdev = test_ensemble_stdev(
                train_data, test_data, device, modalities_order_N, best_weights_synth_N_stdev
            )
                   
            # Test ensemble with ad-hoc weights
            cindex_ensemble_adhoc_N =  test_ensemble(
                train_data, test_data, device, adhoc_weights_N, N,modalities_order_N
            )
        
            # Print results for this number of modalities
            print(f"[Seed {seed}, Fold {fold}] Testing set C-Index (ensemble) with synth weights for {N} modalities: {cindex_ensemble_synth_N:.10f}")
            print(f"[Seed {seed}, Fold {fold}] Testing set C-Index (ensemble) with ad-hoc weights for {N} modalities: {cindex_ensemble_adhoc_N:.10f}")
            print(f"[Seed {seed}, Fold {fold}] Testing set C-Index (ensemble) with synth stdev weights for {N} modalities: {cindex_ensemble_synth_N_stdev:.10f}")
            print("-----")
        
            # Record results
            c_indices_synth[N].append(cindex_ensemble_synth_N)
            c_indices_synth_stdev[N].append(cindex_ensemble_synth_N_stdev)
            c_indices_adhoc[N].append(cindex_ensemble_adhoc_N)

# After processing all seeds and folds, output the results
print('Ensemble C-Indexes for all modality combinations (across all folds and seeds):')
print('\nSynthetic Weights:')
for N in range(2, 7):
    print(f"{N} modalities: C-Indices = {c_indices_synth[N]}")
    avg_cindex_synth_N = sum(c_indices_synth[N]) / len(c_indices_synth[N])
    print(f"Average C-Index for {N} modalities (Synthetic Weights): {avg_cindex_synth_N:.10f}")
    print("-----")
    
print('\nSynthetic Weights(stdev):')
for N in range(2, 7):
    print(f"{N} modalities: C-Indices = {c_indices_synth_stdev[N]}")
    avg_cindex_synth_stdev_N = sum(c_indices_synth_stdev[N]) / len(c_indices_synth_stdev[N])
    print(f"Average C-Index for {N} modalities (Synthetic Weights): {avg_cindex_synth_stdev_N:.10f}")
    print("-----")
     
print('\nAd-Hoc Weights:')
for N in range(2, 7):
    print(f"{N} modalities: C-Indices = {c_indices_adhoc[N]}")
    avg_cindex_adhoc_N = sum(c_indices_adhoc[N]) / len(c_indices_adhoc[N])
    print(f"Average C-Index for {N} modalities (Ad-Hoc stdev Weights): {avg_cindex_adhoc_N:.10f}")
    print("-----")
