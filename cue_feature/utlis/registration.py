import torch

def weighted_procrustes(X, Y, w, eps):
    '''
    X: torch tensor N x 3
    Y: torch tensor N x 3
    w: torch tensor N
    '''

    # https://ieeexplore.ieee.org/document/88573
    assert len(X) == len(Y)
    W1 = torch.abs(w).sum()
    w_norm = w / (W1 + eps)
    mux = (w_norm * X).sum(0, keepdim=True)
    muy = (w_norm * Y).sum(0, keepdim=True)

    # Use CPU for small arrays
    Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()                  # (3, 3) = (3, N) x (N, 3)
    safety_svd = eps * torch.eye(3).to(Sxy)
    # U, D, V = Sxy.svd()
    U, D, V = torch.svd(Sxy + safety_svd)
    S = torch.eye(3).double()
    if U.det() * V.det() < 0:
        S[-1, -1] = -1

    R = U.mm(S.mm(V.t())).float()
    t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
    return R, t


def weighted_procrustes_gpu(X, Y, w, eps):
    '''
    X: torch tensor N x 3
    Y: torch tensor N x 3
    w: torch tensor N
    '''

    # https://ieeexplore.ieee.org/document/88573
    assert len(X) == len(Y)
    W1 = torch.abs(w).sum()
    w_norm = w / (W1 + eps)
    mux = (w_norm * X).sum(0, keepdim=True)
    muy = (w_norm * Y).sum(0, keepdim=True)

    # Use CPU for small arrays
    Sxy = (Y - muy).t().mm(w_norm * (X - mux)).double()                  # (3, 3) = (3, N) x (N, 3)
    safety_svd = eps * torch.eye(3).to(Sxy)
                                                                               # U, D, V = Sxy.svd()
    U, D, V = torch.svd(Sxy + safety_svd)
    S = torch.eye(3).double().to(U)
    if U.det() * V.det() < 0:
        S[-1, -1] = -1

    R = U.mm(S.mm(V.t())).float()
    t = (muy.squeeze() - R.mm(mux.t()).squeeze()).float()
    return R, t
