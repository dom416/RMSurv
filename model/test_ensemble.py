import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model.HFB_fusion_1modal import HFBSurv
from torch.utils.data import DataLoader
from model.utils import CoxLoss, regularize_weights, CIndex_lifeline, cox_log_rank, accuracy_cox,count_parameters
import torch.optim as optim
import pickle
import os
import gc
from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from model.nll_loss_func import NLLSurvLoss
from model.discrete_hazards_plot import plot_survival_probabilities_quartiles, plot_survival_probabilities_modalities
import itertools

import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

#from HFBSurvmain.HFBSurvmain.HFBSurv.model.finetune_uni import custom_collate_fn


def test_ensemble(train_data, test_data, device):
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_case_ids = []
    all_survival_scores = []
        
        
        
    for batch_idx, batch in enumerate(train_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        true_time_bin = batch['true_time_bin'].to(device)
        x_1 = batch['modality_one'].to(device)
        x_2 = batch['modality_two'].to(device)
        x_3 = batch['modality_three'].to(device)
        x_4 = batch['modality_four'].to(device)
        x_5 = batch['modality_five'].to(device)
        #x_6 = batch['modality_six'].to(device)
        x_7 = batch['modality_seven'].to(device)
        
        # find or define mean and standard deviation here:
        
        all_predictions = torch.cat([x_1,x_2,x_3,x_4,x_5,x_7], dim=0)
    
        # Compute the mean and standard deviation along the batch dimension (dimension 0)
        mean_predictions = torch.mean(all_predictions, dim=0)
        std_predictions = torch.std(all_predictions, dim=0)
        mean_predictions = mean_predictions.to(device)
        std_predictions = std_predictions.to(device)
      
      
        # this will normalize all test predictions so that they all have the same mean and std. dev. (derived from the training predictions),
        # This removes the differential effects from overfitting.
    def normalize_predictions(test_pred, mean_predictions, std_predictions):
        # Calculate the mean and standard deviation of the test predictions
        test_pred_mean = torch.mean(test_pred, dim=0)
        test_pred_std = torch.std(test_pred, dim=0)
    
        # Standardize the test predictions to mean 0 and std 1
        standardized_test_pred = (test_pred - test_pred_mean) / test_pred_std
    
        # Scale and shift to match the training set statistics
        normalized_test_pred = standardized_test_pred * std_predictions + mean_predictions
    
        return normalized_test_pred
   
   
    best_mult = 1
    best_ibs = 1
    
    for batch_idx, batch in enumerate(train_loader):
        censor = batch['censor'].to(device).detach().cpu().numpy() 
        survtime = batch['survival_time'].to(device).detach().cpu().numpy() 
        true_time_bin = batch['true_time_bin'].to(device).detach().cpu().numpy()
    
        for std_mult in range(1, 101):
            x_1 = batch['modality_one'].squeeze(1).to(device)
            x_1 = normalize_predictions(x_1, mean_predictions, std_predictions * 0.765 * std_mult).to(device)
            
            x_2 = batch['modality_two'].squeeze(1).to(device)
            x_2 = normalize_predictions(x_2, mean_predictions, std_predictions * 0.23 * std_mult).to(device)
            
            x_3 = batch['modality_three'].squeeze(1).to(device)
            x_3 = normalize_predictions(x_3, mean_predictions, std_predictions * 0.33 * std_mult).to(device)
            
            x_4 = batch['modality_four'].squeeze(1).to(device)
            x_4 = normalize_predictions(x_4, mean_predictions, std_predictions * 0.42 * std_mult).to(device)
            
            x_5 = batch['modality_five'].squeeze(1).to(device)
            x_5 = normalize_predictions(x_5, mean_predictions, std_predictions * 0.315 * std_mult).to(device)
            
            x_7 = batch['modality_seven'].squeeze(1).to(device)
            x_7 = normalize_predictions(x_7, mean_predictions, std_predictions * 0.1 * std_mult).to(device)
    
            hazards1 = torch.sigmoid(x_1)
            survival1 = torch.cumprod(1 - hazards1, dim=1).to(device)
            
            hazards2 = torch.sigmoid(x_2)
            survival2 = torch.cumprod(1 - hazards2, dim=1).to(device)
            
            hazards3 = torch.sigmoid(x_3)
            survival3 = torch.cumprod(1 - hazards3, dim=1).to(device)
            
            hazards4 = torch.sigmoid(x_4)
            survival4 = torch.cumprod(1 - hazards4, dim=1).to(device)
            
            hazards5 = torch.sigmoid(x_5)
            survival5 = torch.cumprod(1 - hazards5, dim=1).to(device)
            
            #hazards6 = torch.sigmoid(x_6)
            #survival6 = torch.cumprod(1 - hazards6, dim=1).to(device)
            
            hazards7 = torch.sigmoid(x_7)
            survival7 = torch.cumprod(1 - hazards7, dim=1).to(device)
            
            # Calculate the mean survival probability
            survival = (survival1+survival2+ survival3 + survival4 + survival5 + survival7) / 6
            
            # Define a dtype for the structured array
            dtype = np.dtype([('event', bool),('time', float)])
        
            # Create structured arrays for training and testing data
            survival_train = np.array(list(zip((1-censor).astype(bool),survtime)), dtype=dtype)
            estimate = survival.detach().cpu().numpy() 
            times = np.arange(1, 21) * (365/2)
            
            #print(survival_train)
            #print(estimate)
            #print(times)
            ibs = integrated_brier_score(survival_train, survival_train, estimate, times)
            
            print(std_mult)
            print(ibs)
            
            if ibs < best_ibs:
                best_ibs = ibs  
                best_mult = std_mult
            print(best_mult)
            print(best_ibs)
   
                
    for batch_idx, batch in enumerate(test_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        true_time_bin = batch['true_time_bin'].to(device)
        x_1 = batch['modality_one'].squeeze(1).to(device)
        x_1_unweighted = normalize_predictions(x_1, mean_predictions, std_predictions).to(device)
        x_1 = normalize_predictions(x_1, mean_predictions, std_predictions*.765*best_mult).to(device)
        x_2 = batch['modality_two'].squeeze(1).to(device)
        x_2_unweighted = normalize_predictions(x_2, mean_predictions, std_predictions).to(device)
        x_2 = normalize_predictions(x_2, mean_predictions, std_predictions*.23*best_mult).to(device)
        x_3 = batch['modality_three'].squeeze(1).to(device)
        x_3_unweighted = normalize_predictions(x_3, mean_predictions, std_predictions).to(device)
        x_3 = normalize_predictions(x_3, mean_predictions, std_predictions*.33*best_mult).to(device)
        x_4 = batch['modality_four'].squeeze(1).to(device)
        x_4_unweighted = normalize_predictions(x_4, mean_predictions, std_predictions).to(device)
        x_4 = normalize_predictions(x_4, mean_predictions, std_predictions*.42*best_mult).to(device)
        x_5 = batch['modality_five'].squeeze(1).to(device)
        x_5_unweighted = normalize_predictions(x_5, mean_predictions, std_predictions).to(device)
        x_5 = normalize_predictions(x_5, mean_predictions, std_predictions*.315*best_mult).to(device) 
        #x_6 = batch['modality_six'].squeeze(1).to(device)
        #x_6 = normalize_predictions(x_6, mean_predictions, std_predictions*.064).to(device)
        x_7 = batch['modality_seven'].squeeze(1).to(device)
        x_7_unweighted = normalize_predictions(x_7, mean_predictions, std_predictions).to(device)
        x_7 = normalize_predictions(x_7, mean_predictions, std_predictions*.1*best_mult).to(device) 
        case_id = batch['case_id']  
    
        # Calculate survival probabilities for each model 
        
        # normalize in this section for each one
        hazards1 = torch.sigmoid(x_1)
        survival1 = torch.cumprod(1 - hazards1, dim=1).to(device)
        
        hazards2 = torch.sigmoid(x_2)
        survival2 = torch.cumprod(1 - hazards2, dim=1).to(device)
        
        hazards3 = torch.sigmoid(x_3)
        survival3 = torch.cumprod(1 - hazards3, dim=1).to(device)
        
        hazards4 = torch.sigmoid(x_4)
        survival4 = torch.cumprod(1 - hazards4, dim=1).to(device)
        
        hazards5 = torch.sigmoid(x_5)
        survival5 = torch.cumprod(1 - hazards5, dim=1).to(device)
        
        #hazards6 = torch.sigmoid(x_6)
        #survival6 = torch.cumprod(1 - hazards6, dim=1).to(device)
        
        hazards7 = torch.sigmoid(x_7)
        survival7 = torch.cumprod(1 - hazards7, dim=1).to(device)
        
        # unweighted calculations only used for plotting
        hazards1 = torch.sigmoid(x_1_unweighted)
        survival1uw = torch.cumprod(1 - hazards1, dim=1).to(device)
        
        hazards2 = torch.sigmoid(x_2_unweighted)
        survival2uw = torch.cumprod(1 - hazards2, dim=1).to(device)
        
        hazards3 = torch.sigmoid(x_3_unweighted)
        survival3uw = torch.cumprod(1 - hazards3, dim=1).to(device)
        
        hazards4 = torch.sigmoid(x_4_unweighted)
        survival4uw = torch.cumprod(1 - hazards4, dim=1).to(device)
        
        hazards5 = torch.sigmoid(x_5_unweighted)
        survival5uw = torch.cumprod(1 - hazards5, dim=1).to(device)
        
        #hazards6 = torch.sigmoid(x_6)
        #survival6 = torch.cumprod(1 - hazards6, dim=1).to(device)
        
        hazards7 = torch.sigmoid(x_7)
        survival7uw = torch.cumprod(1 - hazards7, dim=1).to(device)
        
        # Calculate the mean survival probability
        survival = (survival1+survival2+ survival3 + survival4 + survival5 + survival7) / 6
        
        survival_vectors = [survival.detach().cpu().numpy(),survival1uw.detach().cpu().numpy(),survival2uw.detach().cpu().numpy(),survival3uw.detach().cpu().numpy(),survival4uw.detach().cpu().numpy(),survival5uw.detach().cpu().numpy(),survival7uw.detach().cpu().numpy()]
    
        # Calculate risk 
        risk = 1 - torch.sum(survival, dim=1).detach().cpu().numpy()
        
        all_risk_scores.append(risk)
        all_survival_scores.append(survival.detach().cpu().numpy())
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(survtime.detach().cpu().numpy())
        all_case_ids.append(case_id)
        
        
        survival_test = np.array(list(zip((1-censor.detach().cpu().numpy()).astype(bool),survtime.detach().cpu().numpy())), dtype=dtype) 
        estimate = survival.detach().cpu().numpy() 
        times = np.arange(1, 21) * (365/2)
        ibs_test = integrated_brier_score(survival_train, survival_test, estimate, times)
        
        print('IBS on test set:')
        print(ibs_test)
    
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    
    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    
    directory_path = os.path.join('/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/HFBSurvmain/HFBSurvmain/HFBSurv/results/', 'survival_plots')
    os.makedirs(directory_path, exist_ok=True)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=0, output_dir=directory_path)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=1, output_dir=directory_path)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=2, output_dir=directory_path)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=3, output_dir=directory_path)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=4, output_dir=directory_path)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=5, output_dir=directory_path)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=6, output_dir=directory_path)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=7, output_dir=directory_path)
    plot_survival_probabilities_modalities(survival_vectors, all_censorships, all_event_times, case_index=8, output_dir=directory_path)
    plot_survival_probabilities_quartiles(all_survival_scores, all_censorships, all_event_times, all_case_ids, num_cases=20, output_dir=directory_path)
    
    cindex_test = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return cindex_test
    
    
    
    
#  grid search to find the best weights on the test set (not used for evaluation)    
    


def test_ensemble_gridsearch(train_data, test_data, device):
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=False)
    
    # Load and prepare the training data to calculate mean and std
    for batch_idx, batch in enumerate(train_loader):
        x_1 = batch['modality_one'].to(device)
        x_2 = batch['modality_two'].to(device)
        x_3 = batch['modality_three'].to(device)
        x_4 = batch['modality_four'].to(device)
        x_5 = batch['modality_five'].to(device)
        x_7 = batch['modality_seven'].to(device)
        
        all_predictions = torch.cat([x_1, x_2, x_3, x_4, x_5, x_7], dim=0)
    
        mean_predictions = torch.mean(all_predictions, dim=0)
        std_predictions = torch.std(all_predictions, dim=0)
        mean_predictions = mean_predictions.to(device)
        std_predictions = std_predictions.to(device)
    
    # Load and prepare the test data
    test_data_list = []
    for batch_idx, batch in enumerate(test_loader):
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        true_time_bin = batch['true_time_bin'].to(device)

        x_1 = batch['modality_one'].squeeze(1).to(device)
        x_2 = batch['modality_two'].squeeze(1).to(device)
        x_3 = batch['modality_three'].squeeze(1).to(device)
        x_4 = batch['modality_four'].squeeze(1).to(device)
        x_5 = batch['modality_five'].squeeze(1).to(device)
        x_7 = batch['modality_seven'].squeeze(1).to(device)

        test_data_list.append((censor, survtime, x_1, x_2, x_3, x_4, x_5, x_7))

    def normalize_predictions(test_pred, mean_predictions, std_predictions):
        test_pred_mean = torch.mean(test_pred, dim=0)
        test_pred_std = torch.std(test_pred, dim=0)
        standardized_test_pred = (test_pred - test_pred_mean) / test_pred_std
        normalized_test_pred = standardized_test_pred * std_predictions + mean_predictions
        return normalized_test_pred
    
    weights = [0, 0.25, 0.5, 0.75, 1]
    best_cindex = 0
    best_weights = None
    weight_combinations = list(itertools.product(weights, repeat=6))

    def process_combination(weight_comb):
        risk_scores = []
        censorships = []
        event_times = []
        for censor, survtime, x_1, x_2, x_3, x_4, x_5, x_7 in test_data_list:
            x_1_norm = normalize_predictions(x_1, mean_predictions, std_predictions * weight_comb[0]).to(device)
            x_2_norm = normalize_predictions(x_2, mean_predictions, std_predictions * weight_comb[1]).to(device)
            x_3_norm = normalize_predictions(x_3, mean_predictions, std_predictions * weight_comb[2]).to(device)
            x_4_norm = normalize_predictions(x_4, mean_predictions, std_predictions * weight_comb[3]).to(device)
            x_5_norm = normalize_predictions(x_5, mean_predictions, std_predictions * weight_comb[4]).to(device)
            x_7_norm = normalize_predictions(x_7, mean_predictions, std_predictions * weight_comb[5]).to(device)
            
            hazards1 = torch.sigmoid(x_1_norm)
            survival1 = torch.cumprod(1 - hazards1, dim=1).to(device)
            
            hazards2 = torch.sigmoid(x_2_norm)
            survival2 = torch.cumprod(1 - hazards2, dim=1).to(device)
            
            hazards3 = torch.sigmoid(x_3_norm)
            survival3 = torch.cumprod(1 - hazards3, dim=1).to(device)
            
            hazards4 = torch.sigmoid(x_4_norm)
            survival4 = torch.cumprod(1 - hazards4, dim=1).to(device)
            
            hazards5 = torch.sigmoid(x_5_norm)
            survival5 = torch.cumprod(1 - hazards5, dim=1).to(device)
            
            hazards7 = torch.sigmoid(x_7_norm)
            survival7 = torch.cumprod(1 - hazards7, dim=1).to(device)
        
            survival = (survival1 + survival2 + survival3 + survival4 + survival5 + survival7) / 6
            risk = 1 - torch.sum(survival, dim=1).detach().cpu().numpy()
            
            risk_scores.append(risk)
            censorships.append(censor.detach().cpu().numpy())
            event_times.append(survtime.detach().cpu().numpy())

        risk_scores = np.concatenate(risk_scores)
        censorships = np.concatenate(censorships)
        event_times = np.concatenate(event_times)
        
        cindex_test = concordance_index_censored((1-censorships).astype(bool), event_times, risk_scores, tied_tol=1e-08)[0]
        return weight_comb, cindex_test

    # Automatically use all available CPU cores
    num_threads = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_combination, weight_comb) for weight_comb in weight_combinations]
        for future in as_completed(futures):
            weight_comb, cindex_test = future.result()
            if cindex_test > best_cindex:
                best_cindex = cindex_test
                best_weights = weight_comb

    print(f'Best C-Index: {best_cindex}')
    print(f'Best Weights: {best_weights}')
    return best_cindex, best_weights
    
    


