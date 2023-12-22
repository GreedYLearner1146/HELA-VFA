## HELA-VFA (A HELlinger distance-Attention-based Variational Feature Aggregation Network) ##

This repository contains the relevant codes for our work on `**HELA-VFA: A Hellinger Distance-Attention-based Feature Aggregation
Network for Few-Shot Classification**' (To appear on IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 24)).

**Abstract**: Enabling effective learning using only a few presented examples is a crucial but difficult computer vision objective. Few-shot learning have been proposed to address the
challenges, and more recently variational inference-based approaches are incorporated to enhance few-shot classification performances. However, the current dominant strategy utilized the Kullback-Leibler (KL) divergences to find the log marginal likelihood of the target class distribution, while neglecting the possibility of other probabilistic comparative measures, as well as the possibility of incorporating attention in the feature extraction stages, which can increase the effectiveness of the few-shot model. To this end, we proposed the HELlinger-Attention Variational Feature Aggregation network (HELA-VFA), which utilized the Hellinger distance along with attention in the encoder to fulfill the aforementioned gaps. We show that our approach enables the derivation of an alternate form of the lower bound commonly presented in prior works, thus making the variational optimization feasible and be trained on the same footing in a given setting. Extensive experiments performed on four benchmarked few-shot classification datasets demonstrated the feasibility and superiority of our approach relative to the State-Of-The-Arts (SOTAs) approaches.

We utilized the Sicara few-shot library package for running our few-shot algorithm. Link to the Sicara Few-Shot github page: https://github.com/sicara/easy-few-shot-learning.

All codes here are presented in **PyTorch** format.
