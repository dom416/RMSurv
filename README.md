# RMSurv
Robust Multimodal Survival Model

Abstract: Multimodal cancer survival models have the potential to significantly improve patient outcome prediction and treatment. Many existing methods, however, perform worse or only slightly better than the best unimodal alternative, especially when fusing many modalities. To justify the use of multimodal survival models, they must demonstrate a robust multimodal advantage. In this study, we compare existing methods and introduce several new variations of multimodal late fusion for comparison. RMSurv, or Robust Multimodal Survival Model, our novel discrete model that uses synthetic data generation for time-dependent weight calculation, demonstrates a strong multimodal advantage for the TCGA non-small cell lung cancer datasets and the TCGA pan-cancer dataset. We also demonstrate the effective use of pathology report transformer embeddings for survival modeling and introduce a new normalization method for discrete survival outputs.

For replication, first create a MINDS database (https://github.com/lab-rasool/MINDS) using the provided cohort file or 'TCGA-LUAD'.
We use -omics files downloaded from https://xenabrowser.net/datapages/

For any questions contact dominic flack (daf6674@rit.edu, dominic416@gmail.com)
