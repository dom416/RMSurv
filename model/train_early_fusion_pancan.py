import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from HFBSurvmain.HFBSurvmain.HFBSurv.model.HFB_fusion_1modal import HFBSurv
from torch.utils.data import DataLoader
from HFBSurvmain.HFBSurvmain.HFBSurv.model.utils import CoxLoss, regularize_weights, CIndex_lifeline, cox_log_rank, accuracy_cox,count_parameters
import torch.optim as optim
import pickle
import os
import gc
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from HFBSurvmain.HFBSurvmain.HFBSurv.model.nll_loss_func import NLLSurvLoss
from HFBSurvmain.HFBSurvmain.HFBSurv.model.discrete_hazards_plot import plot_survival_probabilities
from HFBSurvmain.HFBSurvmain.HFBSurv.model.SNN import SeNMo_Trg
#from HFBSurvmain.HFBSurvmain.HFBSurv.model.finetune_uni import custom_collate_fn

####################################################### MFB ############################################################

def train_concat(opt,train_data,test_data,device,num_modalities):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    
    
    if num_modalities == 1:
        model = HFBSurv(38943, (256, 256), (20, 0, 20), (0.05, 0.3), 20, 0.1).to(device)
    if num_modalities == 2:
        model = HFBSurv(39577, (256, 256), (20, 0, 20), (0.05, 0.3), 20, 0.1).to(device)
    if num_modalities == 3:
        model = HFBSurv(39787, (256, 256), (20, 0, 20), (0.05, 0.3), 20, 0.1).to(device)
    if num_modalities == 4:
        model = HFBSurv(232745, (256, 256), (20, 0, 20), (0.05, 0.3), 20, 0.1).to(device)
    if num_modalities == 5:
        model = HFBSurv(232749, (256, 256), (20, 0, 20), (0.05, 0.3), 20, 0.1).to(device)
    if num_modalities == 6:
        model = HFBSurv(236333, (256, 256), (20, 0, 20), (0.05, 0.3), 20, 0.1).to(device)

    
    
    #model = SeNMo_Trg().to(device)
    model.to(device) 
    #model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    train_loader = DataLoader(dataset=train_data, batch_size=200, shuffle=True, drop_last=True)#,pin_memory=True,num_workers=10,collate_fn=custom_collate_fn)
    # batch_size=int(len(train_data)/10)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    c_index_best = 0

    for epoch in tqdm(range(opt.epoch_count, opt.niter+50+1)):
        model.train()
        risk_pred_all, censor_all, survtime_all, case_id_all = np.array([]), np.array([]), np.array([]), np.array([])

        # added in for nll
        all_risk_scores = [] 
        all_censorships = []
        all_event_times = []

        loss_epoch = 0
        gc.collect()
        for batch_idx, batch in enumerate(train_loader):
            censor = batch['censor'].to(device)
            survtime = batch['survival_time'].to(device)
            x_clin = batch['clinical_data'].to(device)
            x_pathr = batch['pathology_report'].to(device)
            x_genex = batch['gene_expression'].to(device)
            x_dnameth = batch['dna_methylation'].to(device)
            x_mirna = batch['mirna'].to(device)
            x_protein = batch['protein_expression'].to(device)
            
            
            if num_modalities == 1:
                fusion = x_dnameth
            if num_modalities == 2:
                fusion = torch.cat((x_dnameth,x_mirna),2)
            if num_modalities == 3:
                fusion = torch.cat((x_dnameth,x_mirna,x_protein),2)
            if num_modalities == 4:
                fusion = torch.cat((x_dnameth,x_mirna,x_protein,x_genex),2)
            if num_modalities == 5:
                fusion = torch.cat((x_dnameth,x_mirna,x_protein,x_genex,x_clin),2)
            if num_modalities == 6:
                fusion = torch.cat((x_dnameth,x_mirna,x_protein,x_genex,x_clin,x_pathr),2)
            

            true_time_bin = batch['true_time_bin'].to(device)
            pred, embeddings = model(fusion) 
            # for SNN only
            #pred = torch.squeeze(pred,1)
        
            # Ensure pred is on the correct device and is a tensor
            pred = pred.to(device)
            
            

            # part for nll loss from PORPOISE:
            hazards = torch.sigmoid(pred)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            #risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            all_risk_scores.append(risk)
            all_censorships.append(censor.detach().cpu().numpy())
            all_event_times.append(survtime.detach().cpu().numpy())
            
            loss_fn = NLLSurvLoss(alpha=0) 
            loss = loss_fn(h=pred, y=true_time_bin, t=survtime, c=censor)
            loss_reg = regularize_weights(model=model)
            loss = loss + opt.lambda_reg *  loss_reg
            loss_epoch += loss.data.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch /= len(train_loader.dataset)

            all_risk_scores = np.concatenate(all_risk_scores)
            all_censorships = np.concatenate(all_censorships)
            all_event_times = np.concatenate(all_event_times)

            cindex_epoch = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

            #pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
            #surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)
            #loss_test, cindex_test, _, _ = test_concat(opt, model, test_data, device,num_modalities)

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            #metric_logger['train']['pvalue'].append(pvalue_epoch)
            #metric_logger['train']['surv_acc'].append(surv_acc_epoch)

            #metric_logger['test']['loss'].append(loss_test)
            #metric_logger['test']['cindex'].append(cindex_test)
            #metric_logger['test']['pvalue'].append(pvalue_test)
            #metric_logger['test']['surv_acc'].append(surv_acc_test)

           # if cindex_test > c_index_best:
            #    c_index_best = cindex_test
           # if opt.verbose > 0:
            print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
            #print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))
        
    return model, optimizer, metric_logger

def test_concat(opt,model, test_data, device,num_modalities):
    model.eval()
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)#,pin_memory=True,num_workers=10,collate_fn=custom_collate_fn)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]),np.array([])
    loss_test = 0
    code_final = None

    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    case_ids = []
    all_embeddings = []

    for batch_idx, batch in enumerate(test_loader):
        case_id = batch['case_id']
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        x_clin = batch['clinical_data'].to(device)
        x_pathr = batch['pathology_report'].to(device)
        x_genex = batch['gene_expression'].to(device)
        x_dnameth = batch['dna_methylation'].to(device)
        x_mirna = batch['mirna'].to(device)
        x_protein = batch['protein_expression'].to(device)

        
        if num_modalities == 1:
            fusion = x_dnameth
        if num_modalities == 2:
            fusion = torch.cat((x_dnameth,x_mirna),2)
        if num_modalities == 3:
            fusion = torch.cat((x_dnameth,x_mirna,x_protein),2)
        if num_modalities == 4:
            fusion = torch.cat((x_dnameth,x_mirna,x_protein,x_genex),2)
        if num_modalities == 5:
            fusion = torch.cat((x_dnameth,x_mirna,x_protein,x_genex,x_clin),2)
        if num_modalities == 6:
            fusion = torch.cat((x_dnameth,x_mirna,x_protein,x_genex,x_clin,x_pathr),2)
                
        true_time_bin = batch['true_time_bin'].to(device)
        
        pred, embeddings = model(fusion) 
        #pred = torch.squeeze(pred,1)
        # Ensure pred is on the correct device and is a tensor
        pred = pred.to(device)
        
        loss_fn = NLLSurvLoss(alpha=0)
        loss = loss_fn(h=pred, y=true_time_bin, t=survtime, c=censor)
        loss_test += loss.data.item()

        # part for nll loss from PORPOISE:
        hazards = torch.sigmoid(pred)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        #risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        all_risk_scores.append(risk)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(survtime.detach().cpu().numpy())
        all_embeddings.append(pred.detach().cpu().numpy())
        case_ids.extend(case_id)

        
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
    
    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)

    
    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_embeddings = np.vstack(all_embeddings)

    cindex_test = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    #pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    #surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    #pred_test = [risk_pred_all, survtime_all, censor_all]
    #code_final_data = code_final.data.cpu().numpy()
    return loss_test, cindex_test, all_embeddings, case_ids
    
  