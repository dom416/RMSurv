import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from HFBSurvmain.HFBSurvmain.HFBSurv.model.HFB_fusion_cox import HFBSurv
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

def train_HFB_cox(opt,train_data,test_data,device):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    model = HFBSurv((38943, 634, 210), (50, 50, 50, 256), (20, 20, 1), (0.1, 0.1, 0.1, 0.3), 20, 0.1).to(device)
    #model = SeNMo_Trg().to(device)
    model.to(device) 
    #model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    train_loader = DataLoader(dataset=train_data, batch_size=len(train_data), shuffle=True, drop_last=True)#,pin_memory=True,num_workers=10,collate_fn=custom_collate_fn)
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
            x_dnameth = batch['dna_methylation'].to(device)
            x_mirna = batch['mirna'].to(device)
            x_protein = batch['protein_expression'].to(device)

            true_time_bin = batch['true_time_bin'].to(device)
            pred, embeddings = model(x_dnameth,x_mirna,x_protein) 

            pred = pred.to(device)
            
            
            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss_reg = regularize_weights(model=model)
            loss = loss_cox + opt.lambda_reg *  loss_reg


            loss_epoch += loss_cox.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch /= len(train_loader.dataset)

            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
            surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)
            loss_test, cindex_test = test_HFB_cox(opt, model, test_data, device)

            print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
            print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))
        
    return model, optimizer, metric_logger

def test_HFB_cox(opt,model, test_data, device):
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
        x_dnameth = batch['dna_methylation'].to(device)
        x_mirna = batch['mirna'].to(device)
        x_protein = batch['protein_expression'].to(device)

        true_time_bin = batch['true_time_bin'].to(device)
        pred, embeddings = model(x_dnameth,x_mirna,x_protein) 
        #pred = torch.squeeze(pred,1)
        # Ensure pred is on the correct device and is a tensor
        pred = pred.to(device)
        loss_cox = CoxLoss(survtime, censor, pred, device)
        loss_test += loss_cox.data.item()
        
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)
    cindex_test =  CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    return loss_test, cindex_test
    
    
