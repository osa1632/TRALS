import torch
import typing
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from get_data import get_data

def init_cores(I_n:typing.List[int], target_ranks:typing.List[int]) -> typing.List[torch.Tensor]:
    N: int = len(target_ranks)
    return [torch.rand((target_ranks[ii-1], I_n[ii], target_ranks[ii])) for ii in range(N)]

def termination_criteria(X:torch.Tensor, cores:typing.List[torch.Tensor], err:torch.Tensor, iter:int):
    show = True#iter%5 == 0
    N: int = len(cores)
    X_hat:torch.Tensor = torch.mm(compute_g_from_cores(cores, N-1), classical_mode_unfolding(cores[N-1],1).transpose(1,0)).reshape(X.shape)

    o = X_hat[1,:,:,:].numpy().squeeze()
    err_new:torch.Tensor = torch.norm(X-X_hat)
    criteria = torch.abs(err_new-err)/err_new

    if show:
        plt.figure(1)
        if iter == 0:
            plt.subplot(121)
            plt.imshow(X[1, :, :, :].numpy().squeeze()+0.5)
            plt.subplot(122)
        plt.imshow(o+0.5)
        plt.title(f'{iter}, {int(criteria*100000)/100000}')
        plt.ion()
        plt.show()
        plt.pause(1e-3)

    err.data = err_new.data
    criteria_met = criteria < 1e-4

    if criteria_met:
        plt.imshow(o+0.5)
        plt.title(f'{iter}, {int(err_new*1000)/1000}')
        plt.show(block=True)

    return criteria_met


def update_core(g, x:torch.Tensor, n) -> torch.Tensor:
    shape_x = x.shape
    N=len(shape_x)
    new_shape = [n, *range(n), *range(n+1,N)]
    x_t = torch.reshape(x.permute(new_shape),(x.shape[0],-1)).T
    return x_t/g


def classical_mode_unfolding(X:torch.Tensor, n:int):
    X_shape = X.shape
    N = len(X_shape)
    perm_vec = [n]+list(range(n))+list(range(n+1,N))
    X_perm = X.permute(perm_vec)
    Xn = X_perm.reshape([X_shape[n], -1])
    return Xn


def mode_unfolding(X:torch.Tensor, n:int):
    X_shape = X.shape
    N = len(X_shape)
    perm_vec = [n]+list(range(n+1,N))+list(range(n))
    X_perm = X.permute(perm_vec)
    Xn = X_perm.reshape([X_shape[n], -1])
    return Xn

def classical_mode_folding(Z:torch.Tensor, n:int, target_shape:torch.Size)->torch.Tensor:
    N:int = len(target_shape)
    perm_vec = list(range(1,n+1))+[0]+list(range(n+1, N))
    target_shape_tuple = [target_shape[n%N]]+[target_shape[ii%N] for ii in range(n)]+[target_shape[ii%N] for ii in range(n+1,N)]
    Z_reshaped:torch.Tensor = torch.reshape(Z, target_shape_tuple)
    X = Z_reshaped.permute(perm_vec)
    return X

def compute_g_from_cores(cores: typing.List[torch.Tensor], n:int) -> torch.Tensor:
    N = len(cores)

    M = classical_mode_unfolding(cores[(n+1)%N], 2).transpose(1,0)

    for jj in range(2,N):
        M = torch.mm(M , classical_mode_unfolding(cores[(n+jj)%N], 0))
        R = cores[(n+jj)%N].shape[2]
        M = M.reshape(-1, R)

    R0, _, R1 = cores[n].shape
    M:torch.Tensor = M.reshape(R1, -1, R0)
    G_mode_2_neq_n = mode_unfolding(M, 1)
    return G_mode_2_neq_n

def norm(ret):
    dim = 0
    M, m = ret.max(dim, keepdim=True)[0], ret.min(dim, keepdim=True)[0]
    # M, m = ret.max(), ret.min()
    # ret = (ret-m)/(M-m+1e-4)-0.5
    choose_Mm = M>torch.abs(m)
    Mm = torch.where(choose_Mm,M,torch.abs(m))
    ret = ret/ (Mm+1e-4)
    return ret

def tr_als(X:torch.Tensor, target_ranks: typing.List[int]) -> typing.List[torch.Tensor]:
    N: int = len(target_ranks)
    cores = init_cores(X.shape, target_ranks)

    err = torch.Tensor([1e+15])
    iter = 0
    mm = torch.mm
    inv = torch.inverse
    t = lambda a: a.transpose(1, 0)
    lstsq = lambda a, b: mm(inv(mm(t(a), a)), mm(t(a), b))

    while not termination_criteria(X, cores, err, iter):
        iter += 1
        for n in range(N):
            cores_complete = compute_g_from_cores(cores, n)
            XnT = mode_unfolding(X, n).transpose(1,0)

            ret = lstsq(cores_complete, XnT).transpose(1,0)

            cores[n] = classical_mode_folding(ret, 1, cores[n].shape)

    else:
        print(iter, err)

    return cores

def get_distribution_core(core:torch.Tensor)->torch.Tensor:
    U = classical_mode_unfolding(core, 1)
    return torch.sum(U**2, 1) / U.shape[1]

def get_distributions(cores:typing.List[torch.Tensor])->typing.List[torch.Tensor]:
    sampling_probs:list = []
    for n in range(len(cores)):
        sampling_probs.append(get_distribution_core(cores[n]))
    return sampling_probs


def tr_als_sampled(X:torch.Tensor, target_ranks: typing.List[int]) -> typing.List[torch.Tensor]:
    N: int = len(target_ranks)
    cores = init_cores(X.shape, target_ranks)

    err = torch.Tensor([1e+15])
    iter = 0
    mm = torch.mm
    inv = torch.inverse
    t = lambda a: a.transpose(1, 0)
    lstsq = lambda a, b: mm(inv(mm(t(a), a)), mm(t(a), b))

    J:int = 100

    while not termination_criteria(X, cores, err, iter):
        iter += 1
        P: list[torch.Tenst] = get_distributions(cores)
        for n in range(N):
            idcs = [list(torch.utils.data.WeightedRandomSampler(P_j, J, True)) for P_j in P]

            cores_sampled = [cores[jj][:,idcs[jj],:] for jj in range(len(cores))]
            X_sampled = X
            for jj in range(len(cores)):
                if jj!=n:
                    X_sampled=torch.index_select(X_sampled, jj, torch.Tensor(idcs[jj]).long())

            cores_complete = compute_g_from_cores(cores_sampled, n)
            XnT = mode_unfolding(X_sampled, n).transpose(1,0)

            ret = lstsq(cores_complete, XnT).transpose(1,0)

            cores[n] = classical_mode_folding(ret, 1, cores[n].shape)
            idcs[n] = torch.utils.data.WeightedRandomSampler(P[n], J, True)

    else:
        print(iter, err)

    return cores


def main():
    # X = torch.Tensor(get_data(("data/coil-100", '100')))-0.5
    X = torch.Tensor(get_data('data/tabby_cat.mp4'))-0.5
    # tr_als(X, [10]*4)
    tr_als_sampled(X, [10]*4)

if __name__ == '__main__':
    main()