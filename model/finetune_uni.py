import os
import gc
import pickle
import torch
from tqdm import tqdm
import random
import timm
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from HFBSurvmain.HFBSurvmain.HFBSurv.model.nll_loss_func import NLLSurvLoss
import torch.optim as optim
from sksurv.metrics import concordance_index_censored, integrated_brier_score
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from HFBSurvmain.HFBSurvmain.HFBSurv.model.utils import CoxLoss, regularize_weights, CIndex_lifeline, cox_log_rank, accuracy_cox
from HFBSurvmain.HFBSurvmain.HFBSurv.model.options import parse_args

####################################################### MFB ############################################################
def custom_collate_fn(batch):
    case_ids = [item['case_id'] for item in batch]
    survival_times = torch.stack([item['survival_time'] for item in batch])
    censors = torch.stack([item['censor'] for item in batch])
    true_time_bins = torch.stack([item['true_time_bin'] for item in batch])
    clinical_data = torch.stack([item['clinical_data'] for item in batch])
    pathology_report = torch.stack([item['pathology_report'] for item in batch])
    uni_slide_image = torch.stack([item['uni_slide_image'] for item in batch])
    
    
    # Determine the max number of patches in the batch
    max_patches = max(item['patches'].shape[0] for item in batch)
    
    # Initialize the padded_patches tensor with zeros
    batch_size = len(batch)
    padded_patches = torch.zeros((batch_size, max_patches, 3, 224, 224))

    for i, item in enumerate(batch):
        patches = item['patches']
        padded_patches[i, :patches.shape[0]] = patches
    return {
        'case_id': case_ids,
        'survival_time': survival_times,
        'censor': censors,
        'patches': padded_patches,
        'true_time_bin': true_time_bins,
        'clinical_data': clinical_data,
        'pathology_report': pathology_report,
        'uni_slide_image': uni_slide_image
    }

#local_dir = "/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/UNI/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"

#model = timm.create_model(
 #   "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, pretrained=False
#)

#checkpoint_path = os.path.join(local_dir, "pytorch_model.bin")
#model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)

#transform = transforms.Compose(
   # [
   #     transforms.Resize(224),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #]
#)

#model.eval()

class MILModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(MILModel, self).__init__()
        self.base_model = base_model
        self.meanpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpooling = nn.AdaptiveMaxPool2d((1, 1)) 
        # Determine the feature size from the base model
        feature_size = self.base_model.head.in_features if hasattr(self.base_model.head, 'in_features') else self.base_model.num_features
        self.fc = nn.Linear(feature_size, num_classes)
    
    def forward(self, x):
        B, T, C, H, W = x.shape  # T: number of patches, C: channels, H: height, W: width
        x = x.view(B * T, C, H, W)  # Reshape to (B*T, C, H, W)
        features = self.base_model.forward_features(x)  # Extract features from each patch
        _, C, F = features.shape
        features = features.view(B, T,C, F)  # Reshape to (B, T, channels, features)
        features = features.permute(0,3,1,2)
        pooled_features = self.meanpooling(features).squeeze(-1).squeeze(-1)  # Pool to (B, features)
        output = self.fc(pooled_features)  # Final classification layer
        return output, pooled_features

# Create an instance of the modified model
#num_classes = 1  # Update with the number of classes you have
#mil_model = MILModel(model, num_classes).cuda()
#mil_model = nn.DataParallel(mil_model, device_ids=[0, 1, 2, 3])

# Base directory for saved data
#base_directory = "/home/daf6674/lambda-stack-with-tensorflow-pytorch/LUAD_fusion/HoneyBee-main/HoneyBee-main/patches_embeddings_labels/"

def finetune_uni(opt,train_dataset):
    cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #mil_model = nn.DataParallel(mil_model, device_ids=[0, 1, 2, 3])
    mil_model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    scaler = GradScaler()
    #print(mil_model)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True,pin_memory=True,num_workers=10,collate_fn=custom_collate_fn)
    num_epochs = 4
    for epoch in range(num_epochs):
        mil_model.train()
        loss_epoch = 0
        risk_pred_all, censor_all, survtime_all, case_id_all = np.array([]), np.array([]), np.array([]), np.array([])
        loss_epoch = 0
        gc.collect()
        for batch_idx, batch in enumerate(train_loader):
            case_id = batch['case_id']
            censor = batch['censor'].cuda()
            survtime = batch['survival_time'].cuda()
            true_time_bin = batch['true_time_bin'].cuda()
            patches = batch['patches'].cuda()
            with autocast():
                pred, testembedding = mil_model(patches)
                loss =CoxLoss(survtime, censor, pred, device)
                loss_reg = regularize_weights(model=model)
                loss = loss + 1e-3 *  loss_reg
            loss_epoch += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            torch.cuda.empty_cache()
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
            torch.cuda.empty_cache()
        loss_epoch /= len(train_loader.dataset)
        cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))   
    return mil_model
 
def get_uni_embeddings(opt,mil_model, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mil_model.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size = 1, shuffle=False)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]),np.array([])
    loss_test = 0
    case_ids = []
    testembeddings = []
    for batch_idx, batch in enumerate(test_loader):
        case_id = batch['case_id']
        censor = batch['censor'].to(device)
        survtime = batch['survival_time'].to(device)
        true_time_bin = batch['true_time_bin'].to(device)
        patches = batch['patches'].to(device)
        pred, testembedding = mil_model(patches)
        loss_test = CoxLoss(survtime, censor, pred, device)
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        case_ids.extend(case_id)  # Extend case_ids list with new case IDs
        testembeddings.append(testembedding.detach().cpu().numpy())
    testembeddings = np.concatenate(testembeddings)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test.item(), 'C-Index', cindex_test.item()))
    return loss_test, cindex_test, testembeddings, case_ids
    
    
def save_uni(embeddings, case_ids, base_dir):
    """
    Save embeddings to corresponding case ID folders.

    Args:
    - embeddings (np.ndarray): The embeddings to save.
    - case_ids (list of str): The case IDs corresponding to each embedding.
    - base_dir (str): The base directory where case ID folders are located.
    """
    for i, case_id in enumerate(case_ids):
        case_dir = os.path.join(base_dir, case_id)
        #os.makedirs(case_dir, exist_ok=True)  # Ensure the directory exists

        embedding_path = os.path.join(case_dir, 'uni_slide_image.npy')
        np.save(embedding_path, embeddings[i])
        
def save_embedding(embeddings, case_ids, base_dir, name):
    """
    Save embeddings to corresponding case ID folders only if the folder exists.

    Args:
    - embeddings (np.ndarray): The embeddings to save.
    - case_ids (list of str): The case IDs corresponding to each embedding.
    - base_dir (str): The base directory where case ID folders are located.
    - name (str): The name of the file to save the embedding as.
    """
    for i, case_id in enumerate(case_ids):
        case_dir = os.path.join(base_dir, case_id)
        if os.path.exists(case_dir):  # Check if the directory exists
            embedding_path = os.path.join(case_dir, name)
            np.save(embedding_path, embeddings[i])
