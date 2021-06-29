
import random
import time

import torch
from matplotlib import pyplot as plt

from get_data import get_data, get_coil_dataset
from tensor_utils import TensorUtils
from trals import TRALS, TRALS_Sampled, randomSVDTR
import coil_test

import unittest
import numpy as np
import pickle

class TRALSTest(unittest.TestCase):
    def test1(self):
        TH = 1e-2
        rank = 10
        max_iters=100
        X = torch.Tensor(get_data("data/coil-100",100)).permute([1,2,3,0])*255
        x_tr = TRALS(X, [rank] * len(X.shape), criteria_th=TH, show=False,
                     termination_criteria_type='rel_error', max_iters=max_iters)
        start_time = time.time()
        x_tr.tr_als()
        stop_time = time.time()
        als_time = stop_time-start_time
        als_index = x_tr.iter_idx
        als_rel_error = x_tr.err
        
        J = 2*rank**2
        
        x_tr = TRALS_Sampled(X, [rank] * len(X.shape), criteria_th=als_rel_error, show=False, J=J,
                                 termination_criteria_type='rel_error', max_iters=als_index, uniform_sampling=False)
        start_time = time.time()
        x_tr.tr_als()
        stop_time = time.time()
        trals_time = stop_time - start_time

        rel_error = x_tr.err
        rals_iter = x_tr.iter_idx
            
        x_tr = randomSVDTR(X, [rank] * len(X.shape), show=False)
        start_time = time.time()
        x_tr.tr_als()
        stop_time = time.time()
        svd_time = stop_time-start_time
        svd_err = x_tr.err
        
        svd_iter = x_tr.iter_idx
        
        print_format = 'time {}, err {}, index{}'
        
        X_hat = TensorUtils.estimate_from_cores(x_tr.cores)
        x_tr.show_random_slice(X_hat, als_rel_error)
        plt.show()
        
        print(f'ALS: {print_format}\nsampALS: {print_format}\nrandSVD: {print_format}\n'.format(als_time, als_rel_error, als_index, trals_time, rel_error, rals_iter , svd_time, svd_err, svd_iter))
            
            
        
    def test6(self):
        TH = 1e-2
        rank = 10
        X = torch.Tensor(get_data("data/coil-100",100)).permute([1,2,3,0])*255
        #X = torch.Tensor(get_data("data/cat.mp4"))
        _,_, *ret = TRALSTest.compare_als_trals(X, rank,TH, 1.010, 1800, show='Red-Truck', svd=True)#1.1 2*rank**2)
        print(ret)
        '''
        X = torch.Tensor(get_data("data/cat.mp4"))*255
        _,*ret = TRALSTest.compare_als_trals(X, rank,TH, 1.1, 800, use_als=True, show='taddy-cat', svd=True)#$
        print(ret)
        '''
        
    @staticmethod
    def compare_als_trals(X, rank=10,TH=1e-3, rel_error_rate=1.1, J_init=200,max_iters=100, show=False, svd=False, use_als=True, **kwargs):
      
        x_tr = TRALS(X, [rank] * len(X.shape), criteria_th=TH, show=False,
                     termination_criteria_type='rel_error', max_iters=max_iters)  ##convergence
        start_time = time.time()
        x_tr.tr_als()
        stop_time = time.time()
        als_time = stop_time-start_time

        als_index = x_tr.iter_idx

        als_rel_error = x_tr.err
        
        if show:
          plt.figure()
          X_hat = TensorUtils.estimate_from_cores(x_tr.cores)
          x_tr.show_random_slice(X_hat, als_rel_error)
          plt.savefig(f'{show}-als.png')

        J = J_init  
        
        
        th_if_use_als = als_rel_error*rel_error_rate if use_als else TH
        while J<5000:
            x_tr = TRALS_Sampled(X, [rank] * len(X.shape), criteria_th=th_if_use_als, show=False, J=J,
                                 termination_criteria_type='rel_error', max_iters=als_index, **kwargs)
            start_time = time.time()
            x_tr.tr_als()
            stop_time = time.time()
            trals_time = stop_time - start_time

            rel_error = x_tr.err
            if show:
                plt.figure()
                X_hat = TensorUtils.estimate_from_cores(x_tr.cores)
                x_tr.show_random_slice(X_hat, rel_error)
                plt.savefig(f'{show}-sketch-als.png')#plt.show(block=True)
            J = J + 1000
        final_J = J
        als_iter = als_index
        rals_iter = x_tr.iter_idx
        
        if svd:
            x_tr = randomSVDTR(X, [rank] * len(X.shape), show=False)
            start_time = time.time()
            x_tr.tr_als()
            stop_time = time.time()
            svd_time = stop_time-start_time
            svd_err = x_tr.err
            X_hat = TensorUtils.estimate_from_cores(x_tr.cores)
        
            if show:
               plt.figure()
               x_tr.show_random_slice(X_hat, svd_err)
               plt.savefig(f'{show}-rSVD.png')#plt.show(block=True)
        else:
            svd_err, svd_time = 0, 0
        return (X_hat,  x_tr, als_rel_error,  rel_error,svd_err,als_time,  trals_time, svd_time, final_J, als_iter, rals_iter)


    def test2(self):
        R = R_tag = 10
        n_cores= 3
        TH = 1e-6
        improvment_factor = 1.01
        J_init = 2*R**2

        TRALSTest.syntetic_data_exp(R_tag,R, TH, n_cores, improvment_factor, J_init)


    def test3(self):
        R_tag = 20
        R = 10
        n_cores= 3
        TH = 1e-4
        improvment_factor = 1.1
        J_init = 2*R**2

        TRALSTest.syntetic_data_exp(R_tag,R, TH, n_cores, improvment_factor, J_init)

    def test5(self):
        R_tag = 20
        R = 10
        n_cores= 3
        TH = 1e-4
        improvment_factor = 1.2
        J_init = 2*R**2
        max_iters=100
        rank = 10
        err_trals,  ts_trals, Js, iters, err_trals_u,  ts_trals_u, Js_u, iters_u = [], [], [], [], [], [], [] ,[]
        Js = [800,2800,4800,5800,7800]#100, 200 ,300,400,500]
        X = torch.Tensor(get_data("data/coil-100",100)).permute([1,2,3,0])*255
        
        x_tr = TRALS(X, [rank] * len(X.shape), criteria_th=TH, show=False,
                     termination_criteria_type='rel_error', max_iters=max_iters)  ##convergence
        start_time = time.time()
        x_tr.tr_als()
        stop_time = time.time()
        als_time = stop_time-start_time
        als_index = x_tr.iter_idx
        als_rel_error = x_tr.err
        
        for J in Js:
          err_trals_,  ts_trals_, Js_, iters_, err_trals_u_,  ts_trals_u_, Js_u_, iters_u_ = [], [], [], [],[],[], [], []
          for _ in range(3):
            x_tr = TRALS_Sampled(X, [rank] * len(X.shape), criteria_th=als_rel_error, show=False, J=J,
                                 termination_criteria_type='rel_error', max_iters=als_index, uniform_sampling=False)
            start_time = time.time()
            x_tr.tr_als()
            stop_time = time.time()
            trals_time = stop_time - start_time

            rel_error = x_tr.err
            rals_iter = x_tr.iter_idx
            
            ts_trals_.append(trals_time)
            err_trals_.append(rel_error)
            iters_.append(rals_iter)
            
            
            x_tr = TRALS_Sampled(X, [rank] * len(X.shape), criteria_th=TH, show=False, J=J,
                                 termination_criteria_type='rel_error', max_iters=als_index, uniform_sampling=True)
            start_time = time.time()
            x_tr.tr_als()
            stop_time = time.time()
            trals_time = stop_time - start_time

            rel_error = x_tr.err
            rals_iter = x_tr.iter_idx
            
            ts_trals_u_.append(trals_time)
            err_trals_u_.append(rel_error)
            iters_u_.append(rals_iter)
        
          err_trals.append((err_trals_))
          ts_trals.append((ts_trals_))
          iters.append((iters_))
        
          err_trals_u.append((err_trals_u_))
          ts_trals_u.append((ts_trals_u_))
          iters_u.append((iters_u_))
        
        
        fig=plt.figure()
        
        ax = fig.add_subplot(1,3,1)
        plt.plot(Js, np.median(err_trals,1), 'bo-', label='sketching')
        plt.plot(Js, np.median(err_trals_u,1), 'r.-',label='uniform-sampling')
       
        plt.title(f'error')
        plt.legend()
        ax.set_yscale('log')


        ax = fig.add_subplot(1,3,2)
        plt.plot(Js, np.median(ts_trals,1), 'b.-', label='sketching')
        plt.plot(Js, np.median(ts_trals_u,1), 'r.-',label='uniform-sampling')
        ax.set_yscale('log')
        
        plt.title('time[sec]')
        plt.legend()
        
        ax = fig.add_subplot(1,3,3)
        plt.plot(Js, np.median(iters,1), 'bo-', label='sketching')
        plt.plot(Js, np.median(iters_u,1), 'r.-',label='uniform-sampling')
        
        plt.title('#iters')
        plt.legend()
        
        
        plt.savefig(f'uniform-sketching.png')
        plt.show()
        pickle.dump([err_trals,  ts_trals, Js, iters, err_trals_u,  ts_trals_u, Js_u, iters_u],open(f'uniform-sketching', 'wb'))
        

    def test4(self):
        rank = 20

        X = torch.Tensor(get_coil_dataset("data/coil-100"))
        test_data = coil_test.main(X)

        x_tr = TRALS_Sampled(X, [rank] * len(X.shape), termination_criteria_type='', J=3800,
                             show=False, max_iters=1)
        x_tr.tr_als()
        X_hat = TensorUtils.estimate_from_cores(x_tr.cores)
        err = x_tr.err
        print(err)
        x_tr.show_random_slice(X_hat, err)
        x_tr.tr_als()

        coil_test.main(X_hat, test_data=test_data)


    @staticmethod
    def syntetic_data_exp(R_tag, R, TH, n_cores, improvment_factor, J_init):
        err_als, err_trals, err_svd, ts_als, ts_trals, ts_svd = [], [], [], [], [], []
        Is = list(range(100, 501,100))
        for I in Is:
          err_als_, err_trals_,err_svd_, ts_als_, ts_trals_ ,ts_svd_= [], [], [], [],[],[]
          for _ in range(3):
            print(I)
            cores = TRALS.init_cores([I] * n_cores, [R_tag] * n_cores)
            for core_idx in range(n_cores):
                cores[core_idx][random.randrange(R_tag), random.randrange(I), random.randrange(R_tag)] = 20.
            synthX = TensorUtils.estimate_from_cores(cores)
            synthX += torch.randn(synthX.shape) * .1

            X_hat,  x_tr, als_rel_error,  rel_error,svd_err,als_time,  trals_time, svd_time, final_J, als_iter, rals_iter = TRALSTest.compare_als_trals(synthX,R,TH, 
                                                                                          improvment_factor, J_init,use_als=True,svd=True,
                                                                                          max_iters=100)

            err_als_.append(als_rel_error)
            err_trals_.append(rel_error)
            err_svd_.append(svd_err)
            ts_als_.append(als_time)
            ts_trals_.append(trals_time)
            ts_svd_.append(svd_time)
            print(als_rel_error,rel_error,svd_err)
          err_als.append(np.median(err_als_))
          err_trals.append(np.median(err_trals_))
          err_svd.append(np.median(err_svd_))
          ts_als.append(np.median(ts_als_))
          ts_trals.append(np.median(ts_trals_))
          ts_svd.append(np.median(ts_svd_))
        fig=plt.figure()
        ax = fig.add_subplot(1,2,1)
        plt.plot(Is, err_als, 'bo-', label='ALS')
        plt.plot(Is, err_svd, 'r*-',label='rSVD')
        plt.plot(Is, err_trals, 'g.-',label='samples-ALS')
       
        plt.title(f'Syntetic-error {R}<->{R_tag}')
        plt.legend()
        ax.set_yscale('log')


        ax=fig.add_subplot(1,2,2)
        plt.plot(Is, ts_als, 'bo-', label='ALS')
        plt.plot(Is, ts_svd, 'r*-',label='rSVD')
        plt.plot(Is, ts_trals, 'g.-',label='samples-ALS')
        plt.title(f'Syntetic-time [sec] {R}<->{R_tag}')
        plt.legend()
        
        ax.set_yscale('log')
        
        plt.savefig(f'syn-{R_tag}-{R}.png')#plt.show()
        
