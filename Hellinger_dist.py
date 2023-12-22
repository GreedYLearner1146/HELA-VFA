import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean

# Function to compute the Hellinger distance. 

_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64

def hellinger(p, q):
    n = p.size(0)
    m = q.size(0)
    d = p.size(1)
    assert d == q.size(1)
    p = p.unsqueeze(1).expand(n, m, d)
    q = q.unsqueeze(0).expand(n, m, d)

    Hell_dists = torch.pow(torch.sqrt(torch.abs(p)) - torch.sqrt(torch.abs(q)),2).sum(2)
    return Hell_dists/ _SQRT2
