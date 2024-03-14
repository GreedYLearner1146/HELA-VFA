## HELA-VFA (A HELlinger distance-Attention-based Variational Feature Aggregation Network) ##

This repository contains the relevant codes for our work on `**HELA-VFA: A Hellinger Distance-Attention-based Feature Aggregation
Network for Few-Shot Classification**' (To appear on IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 24)).

**Abstract**: Enabling effective learning using only a few presented examples is a crucial but difficult computer vision objective. Few-shot learning have been proposed to address the
challenges, and more recently variational inference-based approaches are incorporated to enhance few-shot classification performances. However, the current dominant strategy utilized the Kullback-Leibler (KL) divergences to find the log marginal likelihood of the target class distribution, while neglecting the possibility of other probabilistic comparative measures, as well as the possibility of incorporating attention in the feature extraction stages, which can increase the effectiveness of the few-shot model. To this end, we proposed the HELlinger-Attention Variational Feature Aggregation network (HELA-VFA), which utilized the Hellinger distance along with attention in the encoder to fulfill the aforementioned gaps. We show that our approach enables the derivation of an alternate form of the lower bound commonly presented in prior works, thus making the variational optimization feasible and be trained on the same footing in a given setting. Extensive experiments performed on four benchmarked few-shot classification datasets demonstrated the feasibility and superiority of our approach relative to the State-Of-The-Arts (SOTAs) approaches.

We utilized the Sicara few-shot library package for running our few-shot algorithm. Link to the Sicara Few-Shot github page: https://github.com/sicara/easy-few-shot-learning.

All codes here are presented in **PyTorch** format.

## Table of Results (For miniImageNet) ##

The selected methods are evaluated on the miniImageNet and are based on the approaches by Roy et.al. `FeLMi : Few shot Learning with hard Mixup' [1]. For the tabulated results on CIFAR-FS, tieredImageNet and FC-100, please see our original conference paper. The few-shot evaluation utilized are the 5-way-1-shot and 5-way-5-shot approach. 

| Method | 5-way-1-shot (%) | 5-way-5-shot (%) |
| ------ | ------| ------| 
|ProtoNet| 60.37 $\pm$ 0.83| 78.02 $\pm$ 0.57 |
|TADAM| 58.50 $\pm$ 0.30| 76.70 $\pm$ 0.30 |
|TapNet| 61.65 $\pm$ 0.15| 76.36 $\pm$ 0.10 |
|MetaOptNet| 62.64 $\pm$ 0.61| 78.63 $\pm$ 0.46 |
|MTL| 61.20 $\pm$ 1.80 | 75.50 $\pm$ 0.80 |
|Shot-Free| 59.04 $\pm$ 0.43 | 77.64 $\pm$ 0.39 |
|DSN-MR| 64.60 $\pm$ 0.72 | 79.51 $\pm$ 0.50 |
|Deep-EMD| 65.91 $\pm$ 0.82 | 82.41 $\pm$ 0.56 |
|FEAT| 66.78 $\pm$ 0.20 | 82.05 $\pm$ 0.14 |
|Neg-Cosine| 63.85 $\pm$ 0.81 | 81.57 $\pm$ 0.56 |
|RFS-Simple| 62.02 $\pm$ 0.63 | 79.64 $\pm$ 0.44 |
|RFS-Distill| 64.82 $\pm$ 0.82 | 82.41 $\pm$ 0.43 |
|SKD-Gen1| 66.54 $\pm$ 0.97 | 83.18 $\pm$ 0.54 |
|P-Transfer| 64.21 $\pm$ 0.77 | 80.38 $\pm$ 0.59 |
|MELR| 67.40 $\pm$ 0.43 | 83.40 $\pm$ 0.28 |
|IEPT| 67.05 $\pm$ 0.44 | 82.90 $\pm$ 0.30 |
|IER-distill| 66.85 $\pm$ 0.76 | 84.50 $\pm$ 0.53 |
|Label-Halluc| 67.04 $\pm$ 0.70 | 85.87 $\pm$ 0.48 |
|AssoAlign (ResNet-18)| 59.88 $\pm$ 0.67 | 80.35 $\pm$ 0.73 |
|AssoAlign (WRN-28-10)| 65.92 $\pm$ 0.60 | 82.85 $\pm$ 0.55 |
|FeLMi| 67.47 $\pm$ 0.70 | 86.08 $\pm$ 0.44 |
|**HELA-VFA**| **68.20 $\pm$ 0.30** | **86.70 $\pm$ 0.70** |

Except for AssoAlign, all the methods utilized the ResNet-12 as the training backbone. The backbone for the AssoAlign is stated in the parenthesis.

## Code Instructions ##
The codes instruction presented in this github utilized miniImageNet as an example. For the other datasets, simply download them and change the address path.

1) Run the data_preparation.py, which consists of the library packages, as well as the train-test split.

## Citation Information ##

Please cite the following paper if you find it useful for your work: 

G.Y. Lee, T. Dam, D.P.Poenar, V.N.Duong and M.M. Ferdaus, ``HELA-VFA: A Hellinger Distance-Attention-based Feature Aggregation Network for Few-Shot Classification", in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 2024.

## Some References ##

[1] A. Roy, A. Shah, K. Shah, P. Dhar, A. Cherian, and R. Chellappa, “Felmi: Few shot learning with hard
mixup,” in *Advances in Neural Information Processing Systems*, 2022. 5, 6.
