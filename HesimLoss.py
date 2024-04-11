import torch as F

!pip install pytorch-metric-learning
from pytorch_metric_learning import losses

class Hellinger_Sim_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((torch.sqrt(torch.abs(z_i)), torch.sqrt(torch.abs(z_i))), dim=0)

        # Hellinger Similarity.
        magnitude = (z**2).sum(1).expand(self.batch_size*2, self.batch_size*2)
        sim = F.relu(magnitude).sqrt()

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)
        negative_samples = sim[self.mask]

        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).float()

        logits = torch.cat((positive_samples, negative_samples), dim=0)
        loss = self.criterion(logits[0:100], labels)
        loss /= N

        return loss
