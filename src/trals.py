import random
import typing

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

from tensor_utils import TensorUtils


class TRALS(object):
    def __init__(self, X: torch.Tensor, target_ranks: typing.List[int], criteria_th=1e-2,show=False,
                 max_iters=20, termination_criteria_type='rel_error', is_dynamic_view=False):
        self.X = X
        self.show = show
        self.max_iters = max_iters
        self.X_norm = torch.norm(self.X, 'fro')

        self.target_ranks = target_ranks
        self.cores = TRALS.init_cores(X.shape, target_ranks)
        self.N = len(self.cores)

        self.err = torch.Tensor([1e+15])
        self.criteria_th = criteria_th
        self.termination_criteria_type = termination_criteria_type
        self.is_dynamic_view = is_dynamic_view

    @staticmethod
    def init_cores(I_n: typing.List[int], target_ranks: typing.List[int]) -> typing.List[torch.Tensor]:
        N: int = len(target_ranks)
        return [torch.rand((target_ranks[ii], I_n[ii], target_ranks[(ii+1)%N])) for ii in range(N)]

    def termination_criteria(self, err_new, err_old):
        cost = 1e+100
        type_ = self.termination_criteria_type
        if type_ == 'rel_error':
            cost = err_new
        if type_ == 'convergence':
            cost = (err_old-err_new)/err_old
        return (self.iter_idx >= self.max_iters) or (cost < self.criteria_th)

    def should_stop(self) -> bool:
        X_hat = TensorUtils.estimate_from_cores(self.cores)

        err_new: torch.Tensor = self.get_rel_error(X_hat)
        criteria = err_new#torch.abs(err_new - self.err) / err_new
        # criteria = err_new / np.exp(sum(map(np.log,self.X.shape)))
        criteria_met = self.termination_criteria(err_new=err_new, err_old=self.err)

        if self.show:
            fig=plt.figure(1)
            first_index,_,_=self.show_random_slice(X_hat, criteria)
            if self.is_dynamic_view:
                cid = fig.canvas.mpl_connect('button_press_event',
                                             lambda e: self.show_random_slice(X_hat, criteria,
                                                                              first_index=first_index))

            plt.ion()
            plt.show(block=self.is_dynamic_view)
            plt.pause(1e-3)

        self.err.data = err_new.data

        return criteria_met

    def get_rel_error(self, X_hat):
        return torch.norm(self.X - X_hat, 'fro') / self.X_norm


    def show_random_slice(self, X_hat, criteria,first_index=None):
        self.X/=255.;X_hat/=255.
        if self.X.shape[-1]!=3:
            self.X=self.X.permute([-1,0,1,2])
            X_hat=X_hat.permute([-1,0,1,2])
        if first_index is None:
            first_index = 23#17#random.randrange(self.X.shape[0])
            setattr(self, 'first_index', first_index)
        else:
            self.first_index = (self.first_index+1)% self.X.shape[0]
            first_index = self.first_index
        if len(self.X.shape) > 4:
            last_index = random.randrange(self.X.shape[-1])
        else:
            last_index = None
        o = X_hat[first_index, ..., last_index].numpy().squeeze()

        plt.subplot(121)
        plt.imshow(self.X[first_index, ..., last_index].numpy().squeeze())
        plt.subplot(122)
        plt.imshow(o)

        if not np.isnan(criteria) :
            title = f'{self.iter_idx}, {int(criteria * 100000) / 100000}'
        else:
            title = f'{self.iter_idx}, None'
        plt.title(title)
        self.X*=255.;X_hat*=255.
        return first_index, last_index, o

    def sample_x_cores(self, n):
        cores_complete = TensorUtils.compute_g_from_cores(self.cores, n)
        XnT = TensorUtils.mode_unfolding(self.X, n).transpose(1, 0)
        return XnT, cores_complete

    def per_iter_procedore(self):
        for n in range(self.N):
            XnT, cores_complete = self.sample_x_cores(n)
            self.estimate_core(cores_complete, XnT, n)

    def estimate_core(self, cores_complete, XnT, n):
        ret = TensorUtils.lstsq(cores_complete, XnT).transpose(1, 0)
        self.cores[n] = TensorUtils.classical_mode_folding(ret, 1, self.cores[n].shape)

    def tr_als(self):
        self.iter_idx: int = 0

        for self.iter_idx in tqdm.tqdm(range(self.max_iters)):
            self.iter_idx+=1
            self.per_iter_procedore()
            if self.should_stop():
                break
        # print(iter_idx, self.err)


class TRALS_Sampled(TRALS):
    def __init__(self, X: torch.Tensor, target_ranks: typing.List[int],
                 resample=True,uniform_sampling:bool=False, J:int=200, **kwargs):
        super(TRALS_Sampled, self).__init__(X, target_ranks, **kwargs)
        self.J = J
        self.uniform_sampling = uniform_sampling
        self.resample=resample
        # self.orgJ = J
        self.P: typing.List[torch.Tensor] = self.get_distributions_()
        self.idcs = torch.stack(
            [torch.multinomial(P_jj, self.J, replacement=True) for P_jj in self.P])

    def get_distributions_(self) -> typing.List[torch.Tensor]:
        sampling_probs: list = []
        for n in range(len(self.cores)):
            sampling_probs.append(self.calc_sample_dist(n))
        return sampling_probs

    def calc_sample_dist(self, n):
        if self.uniform_sampling:
            N = self.cores[n].shape[1]
            sampling_prob = torch.ones((N,))/N
        else:
            X = TensorUtils.classical_mode_unfolding(self.cores[n], 1)
            r = torch.matrix_rank(X)
            U, S, VT = torch.svd(X)

            U = U[:, :r]
            sampling_prob = torch.sum(U ** 2, 1) / r
        return sampling_prob

    def per_iter_procedore(self):
        if self.resample:
            idcs = self.idcs

        for n in range(self.N):
            # self.J = random.randrange(int(self.orgJ*0.5),self.orgJ*2)
            # if self.resample:
            #     idcs = self.idcs = torch.stack(
            #         [torch.multinomial(P_jj, self.J, replacement=True) for P_jj in self.P])

            XnT, cores_complete = self.sample_x_cores(n)

            self.estimate_core(cores_complete, XnT, n)

            self.P[n] = self.calc_sample_dist(n)
            # print(self.P[n][0],end=',')
            if self.resample:
                idcs[n].values = torch.multinomial(self.P[n], self.J, replacement=True)
        # print(self.P[0][0])

    def sample_x_cores(self, n):
        if self.resample:
            idcs = self.idcs = torch.stack(
                [torch.multinomial(P_jj, self.J, replacement=True) for P_jj in self.P])

        x_shape = self.X.shape
        # scales = torch.prod(
        #     torch.stack([self.P[jj][idcs[jj]] ** (-0.5) if jj != n else torch.ones_like(self.P[jj][idcs[jj]])
        #                  for jj in range(len(self.cores))], 0), 0).reshape(-1, 1) * (self.J) ** (-0.5*x_shape[0])

        # scales = torch.ones((self.J, 1))

        scales = self.J**(-0.5*self.N)*torch.prod(torch.stack([(self.P[jj][idcs[jj]].reshape(-1,1)**(-0.5) if jj != n else torch.ones((self.J, 1)))
                             for jj in range(self.N)],0),0)

        cores_sampled = [self.cores[jj][:, idcs[jj], :].permute([1, 0,2]) for jj in range(len(self.cores))]
        # cores_sampled = [self.cores[jj][:, idcs[jj], :].permute([1, 0,2]) for jj in range(len(self.cores))]
        cores_complete = scales * TensorUtils.compute_g_from_cores_sampled(cores_sampled, n)
        fix_idcs = TensorUtils.get_absolute_idcs(idcs, n, x_shape)
        X_sampled = torch.stack([x.take(fix_idcs) for x in torch.split(self.X, 1, n)], 1)
        # X_sampled = torch.stack([TensorUtils.take(x, idcs, n) for x in torch.split(self.X, 1, n)], 1)

        XnT = scales * X_sampled
        return XnT, cores_complete

class randomSVDTR(TRALS):
    def __init__(self,X: torch.Tensor, target_ranks: typing.List[int],**kwargs):
        super(randomSVDTR, self).__init__(X, target_ranks, **kwargs)
        self.oversamp = 10
        self.max_iters = 1

    def per_iter_procedore(self):
        sz = list(self.X.shape)
        C = self.X.reshape(sz[0], -1)
        Z = C.matmul( torch.randn(np.prod(sz[1:]), self.target_ranks[-1] * self.target_ranks[0] + self.oversamp))

        Q, _, _ = torch.svd(Z)
        Q = Q[ :,:self.target_ranks[-1]*self.target_ranks[0]]
        self.cores[0] = Q.reshape([ sz[0],self.target_ranks[-1], self.target_ranks[0]]).permute( [1,0,2])
        C_ten = (Q.T.matmul(C)).reshape([Q.shape[1], *sz[1:]])

        C_ten = C_ten.reshape([self.target_ranks[-1], self.target_ranks[0], -1])
        C_ten = C_ten.permute([1,2,0])
        C_ten = C_ten.reshape([self.target_ranks[0] * sz[1], -1, self.target_ranks[-1]])

        for n in range(1,self.N-1):
            C = C_ten.reshape([self.target_ranks[n - 1] * sz[n], -1])
            Z = C.matmul(torch.randn(int(np.prod(sz[n + 1:]))*self.target_ranks[-1], self.target_ranks[n] + self.oversamp))
            Q, _,_ = torch.svd(Z)
            Q = Q[:, :self.target_ranks[n]]
            self.cores[n] = Q.reshape([self.target_ranks[n - 1], sz[n], self.target_ranks[n]])
            C_ten = Q.T.matmul(C).reshape([self.target_ranks[n], -1, self.target_ranks[-1]])
        self.cores[-1] = C_ten
