This repository contains the relevant codes for our work on `HELA-VFA: A Hellinger Distance-Attention-based Feature Aggregation
Network for Few-Shot Classification' (To appear on IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 24)).

Abstract: Enabling effective learning using only a few presented examples is a crucial but difficult computer vision objective. Few-shot learning have been proposed to address the
challenges, and more recently variational inference-based approaches are incorporated to enhance few-shot classification performances. However, the current dominant
strategy utilized the Kullback-Leibler (KL) divergences to find the log marginal likelihood of the target class distribution, while neglecting the possibility of other probabilistic comparative measures, as well as the possibility
of incorporating attention in the feature extraction stages, which can increase the effectiveness of the few-shot model. To this end, we proposed the HELlinger-Attention Variational Feature Aggregation network (HELA-VFA),
which utilized the Hellinger distance along with attention in the encoder to fulfill the aforementioned gaps. We show that our approach enables the derivation of an alternate form of the lower bound commonly presented in
prior works, thus making the variational optimization feasible and be trained on the same footing in a given setting. Extensive experiments performed on four benchmarked few-shot classification datasets demonstrated the
feasibility and superiority of our approach relative to the State-Of-The-Arts (SOTAs) approaches.
