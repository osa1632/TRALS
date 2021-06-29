import typing

import torch


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