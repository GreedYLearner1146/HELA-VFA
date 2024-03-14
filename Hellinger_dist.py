import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean

def Hellinger_dist(x,y): # Where N is the number of class.
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x1 = x.unsqueeze(1).expand(n, m, d)
    y1 = y.unsqueeze(0).expand(n, m, d)

    _SQRT2 = np.sqrt(2)

    # Inputs for variational computation. Use mean and std of the given inputs.
    x_m = torch.mean(x1)
    y_m = torch.mean(y1)
    x_std = torch.mean(x1)
    y_std  = torch.mean(y1)

    # The reparameterization trick. The second half of the parenthesis is the normal distribution.
    P1 = x_m + x_std*(1/(torch.sqrt(torch.abs(2*np.pi*x_std*x_std))))*torch.exp(-((x1-x_m)*(x1-x_m))/(2*x_std*x_std))
    Q1 = y_m + y_std*(1/(torch.sqrt(torch.abs(2*np.pi*y_std*y_std))))*torch.exp(-(y1-y_m)*(y1-y_m)/(2*y_std*y_std))
    Hell_dists = torch.pow(torch.sqrt(torch.abs(P1)) - torch.sqrt(torch.abs(Q1)),2).sum(2)

    return Hell_dists/ _SQRT2
