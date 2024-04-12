####################################################################################

class SqRootSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        return torch.matmul(query_emb, ref_emb.t())

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.sum(query_emb * ref_emb, dim=1)

####################################################################################

class HellingerSimilarity(SqRootSimilarity):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=True, **kwargs)
        assert self.is_inverted
        assert self.normalize_embeddings

####################################################################################

class HesimLoss(GenericPairLoss):
    def __init__(self, temperature=0.01, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(torch.sqrt(torch.abs(pos_pairs)) - torch.sqrt(torch.abs(max_val))).squeeze(1)   # Hellinger similarity.
            denominator = torch.sum(torch.exp(torch.sqrt(torch.abs(neg_pairs)) - torch.sqrt(torch.abs(max_val))), dim=1) + numerator  # Hellinger similarity.
            log_exp = torch.log((numerator / denominator) + small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

    def get_default_distance(self):
        return HellingerSimilarity()

#########################################################################
Hesimloss= HesimLoss(temperature=0.01)  # Pre-requisite to running the HesimLoss criterion in the training code model_train.py.
