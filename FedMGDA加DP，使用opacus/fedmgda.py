from flgo.algorithm.fedbase import BasicServer, BasicClient
import torch
from flgo.utils import fmodule
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import os
import flgo
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import flgo.experiment.analyzer
"""
This is a non-official implementation of 'FedMGDA+: Federated Learning meets Multi-objective Optimization' (http://arxiv.org/abs/2006.11489)
"""
import torch

from flgo.utils import fmodule
from flgo.algorithm.fedbase import BasicServer
from flgo.algorithm.fedavg import Client
import numpy as np
import copy
import cvxopt

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'eta':1, 'epsilon':0.1})

    def aggregate(self, models: list, *args, **kwargs):
        # calculate normalized gradients
        grads = [self.model - w for w in models]
        for i in range(len(grads)):
            gi_norm = 0.0
            for p in grads[i].parameters():
                gi_norm += (p**2).sum()
            grads[i] = grads[i]/(torch.sqrt(gi_norm) + 1e-8)
        # for gi in grads: gi.normalize()
        # calculate λ0
        nks = [len(self.clients[cid].train_data) for cid in self.received_clients]
        nt = sum(nks)
        lambda0 = [1.0 * nk / nt for nk in nks]
        # optimize lambdas to minimize ||λ'g||² s.t. λ∈Δ, ||λ - λ0||∞ <= ε
        op_lambda = self.optim_lambda(grads, lambda0)
        op_lambda = [ele[0] for ele in op_lambda]
        # aggregate grads
        dt = fmodule._model_average(grads, op_lambda)
        return self.model - dt * self.eta

    def optim_lambda(self, grads, lambda0):
        # create H_m*m = 2J'J where J=[grad_i]_n*m
        n = len(grads)
        Jt = []
        for gi in grads:
            Jt.append((copy.deepcopy(fmodule._modeldict_to_tensor1D(gi.state_dict())).cpu()).numpy())
        Jt = np.array(Jt)
        # target function
        P = 2 * np.dot(Jt, Jt.T)

        q = np.array([[0] for i in range(n)])
        # equality constraint λ∈Δ
        A = np.ones(n).T
        b = np.array([1])
        # boundary
        lb = np.array([max(0, lambda0[i] - self.epsilon) for i in range(n)])
        ub = np.array([min(1, lambda0[i] + self.epsilon) for i in range(n)])
        G = np.zeros((2*n,n))
        for i in range(n):
            G[i][i]=-1
            G[n+i][i]=1
        h = np.zeros((2*n,1))
        for i in range(n):
            h[i] = -lb[i]
            h[n+i] = ub[i]
        res=self.quadprog(P, q, G, h, A, b)
        return res

    def quadprog(self, P, q, G, h, A, b):
        """
        Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
        Output: Numpy array of the solution
        """
        P = cvxopt.matrix(P.tolist())
        q = cvxopt.matrix(q.tolist(), tc='d')
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist(), tc='d')
        sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
        return np.array(sol['x'])
class Client(BasicClient):
    def get_batch_data(self, data_loader):
        """
        Get the batch of training data
        Returns:
            a batch of data
        """
        if self._train_loader is None:
            self._train_loader = data_loader
        try:
            batch_data = next(data_loader)
        except Exception as e:
            data_loader = iter(self._train_loader)
            batch_data = next(data_loader)
        # clear local_movielens_recommendation DataLoader when finishing local_movielens_recommendation training
        self.current_steps = (self.current_steps + 1) % self.num_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data
    # @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        data_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.loader_num_workers,
            pin_memory=self.option['pin_memory'],
            drop_last=not self.option.get('no_drop_last', False)
        )
        MAX_GRAD_NORM = 1.2
        EPSILON = 50
        DELTA = 1e-5
        EPOCHS = 5
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            # epochs=EPOCHS,
            # target_epsilon=EPSILON,
            # target_delta=DELTA,
            # max_grad_norm=MAX_GRAD_NORM,
        )


        # for iter in range(self.num_steps):
        #     # get a batch of data
        #     batch_data = data_loader
        #     model.zero_grad()
        #     # calculate the loss of the model on batched dataset through task-specified calculator
        #     loss = self.calculator.compute_loss(model, batch_data)['loss']
        #     loss.backward()
        #     if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
        #     optimizer.step()
        # return
        # for iter in range(self.num_steps):
        #     for batch_data in data_loader:
        #         inputs, labels = batch_data
        #         model.zero_grad()
        #         optimizer.zero_grad()
        #         # 计算模型在批数据上的损失
        #         loss = self.calculator.compute_loss(model, (inputs, labels))['loss']
        #         loss.backward()
        #         if self.clip_grad > 0:
        #             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
        #         optimizer.step()
        # return
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data(data_loader)
            model.zero_grad()
            optimizer.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return
class fedmgda:
    Server = Server
    Client = Client

task = './my_task1'
flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=10), task)
runner = flgo.init(task, fedmgda, option={'num_rounds':100, 'local_test':True, 'learning_rate':0.005, })
runner.run()
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fedmgda']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)