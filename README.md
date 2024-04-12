## HELA-VFA (A HELlinger distance-Attention-based Variational Feature Aggregation Network) ##

This repository contains the relevant codes for our work on `**HELA-VFA: A Hellinger Distance-Attention-based Feature Aggregation
Network for Few-Shot Classification**' (Appeared on IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 24)).

**Abstract**: Enabling effective learning using only a few presented examples is a crucial but difficult computer vision objective. Few-shot learning have been proposed to address the
challenges, and more recently variational inference-based approaches are incorporated to enhance few-shot classification performances. However, the current dominant strategy utilized the Kullback-Leibler (KL) divergences to find the log marginal likelihood of the target class distribution, while neglecting the possibility of other probabilistic comparative measures, as well as the possibility of incorporating attention in the feature extraction stages, which can increase the effectiveness of the few-shot model. To this end, we proposed the HELlinger-Attention Variational Feature Aggregation network (HELA-VFA), which utilized the Hellinger distance along with attention in the encoder to fulfill the aforementioned gaps. We show that our approach enables the derivation of an alternate form of the lower bound commonly presented in prior works, thus making the variational optimization feasible and be trained on the same footing in a given setting. Extensive experiments performed on four benchmarked few-shot classification datasets demonstrated the feasibility and superiority of our approach relative to the State-Of-The-Arts (SOTAs) approaches.

We utilized the Sicara few-shot library package for running our few-shot algorithm. Link to the Sicara Few-Shot github page: https://github.com/sicara/easy-few-shot-learning.

All codes here are presented in **PyTorch** format.

The link to our paper can be found at https://openaccess.thecvf.com/content/WACV2024/papers/Lee_HELA-VFA_A_Hellinger_Distance-Attention-Based_Feature_Aggregation_Network_for_Few-Shot_Classification_WACV_2024_paper.pdf 

(**This repo is still updating as of current. Stay tune for latest changes.**)

## Preliminary Results (For miniImageNet) ##

The selected methods (in the paper) are evaluated on the miniImageNet and are based on the approaches by Roy et.al. `FeLMi : Few shot Learning with hard Mixup' [1]. For the tabulated results on CIFAR-FS, tieredImageNet and FC-100, please see our original conference paper. The few-shot evaluation utilized are the 5-way-1-shot and 5-way-5-shot approach. 

| Method | 5-way-1-shot (%) | 5-way-5-shot (%) |
| ------ | ------| ------| 
|**HELA-VFA**| **68.2 $\pm$ 0.30** | **86.7 $\pm$ 0.70** |

The backbone for our HELA-VFA is the ResNet-12.

## Code Instructions ##
The codes instructions presented in this github utilized miniImageNet as an example. For the other datasets, simply download them and change the address path.

1) Run the Data_Preparation.py, which consists of the library packages, as well as the train-test split. The casting of the images into the respective array format are also performed.
2) Run the Data_Augmentation.py, which contains the essential transformations for data augmentation. These include horizontal and vertical flips, as well as rotations by 90 and 270 degrees, all at a probability of 0.5 (p = 0.5).
3) Run the Attention.py, which comprises the channel and spatial attention module.
4) Run the ResNet12.py, which contains the Resnet-12 backbone (with attention module incorporated).
5) Run the dataloader.py, which contains the dataloader for the training and testing set.
6) Run the Hellinger_dist.py
7) Run the HELA_VFA_main.py
8) Run test_sampler_loader.py, which contains the hyperparameters N_SHOT, N_WAY, N_QUERY, and number of evaluation task which can be easily configured.
9) Run the sub-functions contained in the folder Hesim, which comprise the codes for the various helper functions leading up to the Hesim loss function as highlighted in our paper. The helper functions are mainly adapted from the pytorch metric learning library by Kevin Musgrave: https://github.com/KevinMusgrave/pytorch-metric-learning. 
10) Run model_train.py to train the model.
11) Finally, evaluate.py to see how the model performed before the training.

## Citation Information ##

Please cite the following paper if you find it useful for your work: 

G.Y. Lee, T. Dam, D.P.Poenar, V.N.Duong and M.M. Ferdaus, ``HELA-VFA: A Hellinger Distance-Attention-based Feature Aggregation Network for Few-Shot Classification", in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 2024.

## Some References ##

[1] A. Roy, A. Shah, K. Shah, P. Dhar, A. Cherian, and R. Chellappa, “Felmi: Few shot learning with hard
mixup,” in *Advances in Neural Information Processing Systems*, 2022. 5, 6.
