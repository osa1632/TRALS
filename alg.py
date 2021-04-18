import torch
import typing
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
import tqdm

from get_data import get_data
torch.random.manual_seed(42)

class TensorUtils:
    @staticmethod
    def classical_mode_unfolding(X: torch.Tensor, n: int):
        X_shape = X.shape
        N = len(X_shape)
        perm_vec = [n] + [ii%N for ii in range(N) if ii!=n]
        X_perm:torch.Tensor = X.permute(perm_vec)
        Xn = X_perm.contiguous().view([X_shape[n], -1])
        return Xn

    @staticmethod
    def mode_unfolding(X: torch.Tensor, n: int):
        X_shape = X.shape
        N = len(X_shape)
        perm_vec = [n] + [(ii+n)%N for ii in range(1, N)]
        X_perm = X.permute(perm_vec)
        Xn = X_perm.reshape([X_shape[n], -1])
        return Xn

    @staticmethod
    def classical_mode_folding(Z: torch.Tensor, n: int, target_shape: torch.Size) -> torch.Tensor:
        N: int = len(target_shape)
        perm_vec = list(range(1, n + 1)) + [0] + list(range(n + 1, N))
        target_shape_tuple = [target_shape[n]] + \
                             [target_shape[ii] for ii in range(N) if ii!=n]

        Z_reshaped: torch.Tensor = Z.view(target_shape_tuple)
        X = Z_reshaped.permute(perm_vec)
        return X

    @staticmethod
    def norm(ret):
        dim = 0
        M, m = ret.max(dim, keepdim=True)[0], ret.min(dim, keepdim=True)[0]

        choose_Mm = M > torch.abs(m)
        Mm = torch.where(choose_Mm, M, torch.abs(m))
        ret = ret / (Mm + 1e-4)

        return ret

    @staticmethod
    def lstsq(a: torch.Tensor, b: torch.Tensor):
        return torch.pinverse(a).mm(b)
        # return torch.Tensor(np.linalg.pinv(a).dot(b))
        # mm = torch.mm
        # inv = torch.inverse
        # t = lambda a: a.transpose(1, 0)
        # return mm(inv(mm(t(a), a)), mm(t(a), b))

    @staticmethod
    def get_absolute_idcs(idcs, n, x_shape):
        x_shape_dims_wo_n = [1] + [dim for ii, dim in enumerate(x_shape) if ii != n][1:][::-1]
        product = torch.cumprod(torch.Tensor(x_shape_dims_wo_n), dim=0).long().reshape(1, -1).flip(1)
        idcs_wo_n = torch.cat([idcs[:n,:], idcs[n + 1:,:]],0)
        fix_idcs = torch.mm(product, idcs_wo_n).squeeze()
        return fix_idcs

    @staticmethod
    def estimate_from_cores(cores):
        N: int = len(cores)
        X_hat: torch.Tensor = torch.mm(TensorUtils.compute_g_from_cores(cores, N - 1),
                                       TensorUtils.classical_mode_unfolding(cores[N - 1], 1).transpose(1,
                                                                                                            0)).reshape(
            [c.shape[1] for c in cores])
        return X_hat
    # def estimate_from_cores(cores):
    #     N: int = len(cores)
    #     X_hat: torch.Tensor = torch.mm(TensorUtils.compute_g_from_cores(cores, N - 1),
    #                                    TensorUtils.classical_mode_unfolding(cores[N - 1], 1).transpose(1,0)).reshape(
    #         [c.shape[0] for c in cores])
    #     return X_hat

    @staticmethod
    def compute_g_from_cores(cores: typing.List[torch.Tensor], n: int) -> torch.Tensor:
        N = len(cores)

        M = TensorUtils.classical_mode_unfolding(cores[(n + 1) % N], 2).transpose(1, 0)

        for jj in range(2, N):
            M = torch.mm(M, TensorUtils.classical_mode_unfolding(cores[(n + jj) % N], 0))
            R = cores[(n + jj) % N].shape[2]
            M = M.reshape(-1, R)

        R0, _, R1 = cores[n].shape
        M: torch.Tensor = M.reshape(R1, -1, R0)
        G_mode_2_neq_n = TensorUtils.mode_unfolding(M, 1)
        return G_mode_2_neq_n

    @staticmethod
    def compute_g_from_cores_sampled(cores: typing.List[torch.Tensor], n: int) -> torch.Tensor:
        N = len(cores)
        mode_unfolding_idx = [(n + ii) % N for ii in range(2, N)]

        mat = cores[(n + 1) % N]

        for idx in mode_unfolding_idx:
            permuted_core = cores[idx]
            mat = torch.matmul(mat, permuted_core)

        # mat = mat.permute([2,0,1])#1,2,0
        mat = mat.permute([1, 0, 2])  # 1,0,2
        # mat = mat.permute([1,0,2])#1,0,2
        # mat = mat.permute([1,0,2])#1,2,0
        mat = TensorUtils.mode_unfolding(mat, 1)

        return mat

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
            self.is_dynamic_view=self.iter_idx > 20
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
        plt.imshow(self.X[first_index, ..., last_index].numpy().squeeze().astype('uint8'))
        plt.subplot(122)
        plt.imshow(o.astype('uint8'))
        if not np.isnan(criteria) :
            title = f'{self.iter_idx}, {int(criteria * 100000) / 100000}'
        else:
            title = f'{self.iter_idx}, None'
        plt.title(title)

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
            if self.should_stop():
                break
            self.iter_idx+=1
            self.per_iter_procedore()

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


def example():
    # X = torch.Tensor(get_data(("data/tabby_cat.mp4")))/255.-0.5
    TH=1e-2
    rank=20
    # x_1=0
    x_1=1
    if x_1:
        X = torch.Tensor(get_data("data/coil-100",100)) ;f=1
        # X = torch.Tensor(get_data("data/coil-100")) ;f=1
    # else:
    #     X = torch.Tensor(range(2 * 3 * 4 * 5)).reshape([2,3,4,5]);f=0
    # X = torch.Tensor(get_data(("data/coil-100", 'obj100'))) - 0.5
    # x_tr = TRALS(X, [rank] * len(X.shape), criteria_th=TH, show=True)
    # x_tr = TRALS_Sampled(X, [10] * len(X.shape))
    # x_tr = TRALS(X, [10] * 4)
    # x_tr.tr_als()
    # print(x_tr.iter_idx, x_tr.err)
    x_tr = TRALS(X, [rank] * len(X.shape),termination_criteria_type='',#J=5800,
                         show=1,max_iters=1000)#, J=200)criteria_th=TH,
    x_tr.tr_als()

    # print(x_tr.iter_idx, x_tr.err)
    print(X[:,0,0,1])
    raise Exception

def exp1():
    TH = 1e-3
    rank = 10
    X = torch.Tensor(get_data("data/coil-100", 100)) - 0.5

    X_hat, rel_error, x_tr, als_erl_err = compare_als_trals(TH, X, rank, 1.1, 2*rank**2)
    x_tr.show_random_slice(X_hat, rel_error)
    plt.show(block=True)


def compare_als_trals(TH, X, rank, rel_error_rate=1.1, J_init=200,max_iters=100):
    x_tr = TRALS(X, [rank] * len(X.shape), criteria_th=TH, show=False,
                 termination_criteria_type='rel_error', max_iters=100)  ##convergence
    x_tr.tr_als()
    X_hat = TensorUtils.estimate_from_cores(x_tr.cores)
    als_index = x_tr.iter_idx
    # 20: 100 0.0552
    # 10: 100 4200 tensor(0.1601) [100 tensor(0.1495)]
    # x_tr.show_random_slice(X_hat, rel_error)
    # plt.show(block=True)
    als_rel_error = x_tr.get_rel_error(X_hat)
    print(x_tr.iter_idx, als_rel_error)
    J = J_init  # 10800
    rel_error = 1e+100
    while rel_error / als_rel_error >= rel_error_rate:
        x_tr = TRALS_Sampled(X, [rank] * len(X.shape), criteria_th=TH, show=False, J=J,
                             termination_criteria_type='', max_iters=als_index)
        x_tr.tr_als()
        X_hat = TensorUtils.estimate_from_cores(x_tr.cores)
        rel_error = x_tr.get_rel_error(X_hat)
        print(rel_error, rel_error / als_rel_error, J)
        J = J + 1000
    print(x_tr.iter_idx, J, rel_error)
    return X_hat, rel_error, x_tr, als_rel_error


def exp2():
    R_tag = 10
    n_cores= 3
    TH = 1e-6

    err_als, err_trals = [] , []
    Is = range(100,501,100)
    for I in Is:
        cores = TRALS.init_cores([I]*n_cores, [R_tag]*n_cores)
        for core_idx in range(n_cores):
            cores[core_idx][random.randrange(R_tag),random.randrange(I),random.randrange(R_tag)]=20.
        synthX = TensorUtils.estimate_from_cores(cores, [R_tag]*3)
        synthX+=torch.randn(synthX.shape)*.1

        X_hat, rel_error, x_tr, als_erl_err = compare_als_trals(TH, synthX, R_tag, 1.2, 200,max_iters=500)
        err_als.append(als_erl_err)
        err_trals.append(rel_error)
    plt.plot(Is, err_als, 'bo')
    plt.plot(Is, err_trals, 'g*')
    plt.show()


def get_args():
    parser = argparse.ArgumentParser('sampling')
    parser.add_argument('--input_path', default='small')
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--csv_result', default='f_nf.csv')
    parser.add_argument('--manually_flip_selection', action='store_true')
    parser.add_argument('--features_extraction', choices=[None, 'vgg', 'pca'], default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

def main():
    example()
    # exp1()
    exp2()

if __name__ == '__main__':
    main()
