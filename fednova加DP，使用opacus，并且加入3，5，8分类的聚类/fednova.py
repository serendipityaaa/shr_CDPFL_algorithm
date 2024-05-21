"""
This is a non-official implementation of 'Tackling the Objective Inconsistency Problem
in Heterogeneous Federated Optimization' (http://arxiv.org/abs/2007.07481)
"""
from flgo.algorithm.fedbase import BasicServer, BasicClient
from flgo.utils import fmodule
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

# class Server(BasicServer):
#     def initialize(self, *args, **kwargs):
#         self.sample_option = 'uniform'
#
#     def iterate(self):
#         self.selected_clients = self.sample()
#         # training
#         res = self.communicate(self.selected_clients)
#         models, taus = res['model'], res['tau']
#         ds = [(model-self.model)/tauk for model, tauk in zip(models, taus)]
#         local_data_vols = [c.datavol for c in self.clients]
#         total_data_vol = sum(local_data_vols)
#         self.model = self.aggregate(ds, taus, p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients])
#         return
#
#     def aggregate(self, ds, taus, p=[]):
#         if not ds: return self.model
#         if self.aggregation_option == 'weighted_scale':
#             K = len(ds)
#             N = self.num_clients
#             tau_eff = sum([tauk*pk for tauk,pk in zip(taus, p)])
#             delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)]) * N / K
#         elif self.aggregation_option == 'uniform':
#             tau_eff = 1.0*sum(taus)/len(ds)
#             delta = fmodule._model_average(ds)
#
#         elif self.aggregation_option == 'weighted_com':
#             tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
#             delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)])
#         else:
#             sump = sum(p)
#             p = [pk/sump for pk in p]
#             tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
#             delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)])
#         return self.model + tau_eff * delta
#
# class Client(BasicClient):
#     def pack(self, model):
#         tau = self._working_amount if hasattr(self, '_working_amount') else self.num_steps
#         return {
#             "model" : model,
#             "tau": tau,
#         }
#     def get_batch_data(self, data_loader):
#         """
#         Get the batch of training data
#         Returns:
#             a batch of data
#         """
#         if self._train_loader is None:
#             self._train_loader = data_loader
#         try:
#             batch_data = next(data_loader)
#         except Exception as e:
#             data_loader = iter(self._train_loader)
#             batch_data = next(data_loader)
#         # clear local_movielens_recommendation DataLoader when finishing local_movielens_recommendation training
#         self.current_steps = (self.current_steps + 1) % self.num_steps
#         if self.current_steps == 0:
#             self.data_loader = None
#             self._train_loader = None
#         return batch_data
#     # @fmodule.with_multi_gpus
#     def train(self, model):
#         model.train()
#         optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
#                                                   momentum=self.momentum)
#         data_loader = DataLoader(
#             self.train_data,
#             batch_size=self.batch_size,
#             num_workers=self.loader_num_workers,
#             pin_memory=self.option['pin_memory'],
#             drop_last=not self.option.get('no_drop_last', False)
#         )
#         MAX_GRAD_NORM = 1.2
#         EPSILON = 50
#         DELTA = 1e-5
#         EPOCHS = 5
#         privacy_engine = PrivacyEngine()
#         model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
#             module=model,
#             optimizer=optimizer,
#             data_loader=data_loader,
#             # noise_multiplier=1.0,
#             # max_grad_norm=1.0,
#             epochs=EPOCHS,
#             target_epsilon=EPSILON,
#             target_delta=DELTA,
#             max_grad_norm=MAX_GRAD_NORM,
#         )
#
#
#         # for iter in range(self.num_steps):
#         #     # get a batch of data
#         #     batch_data = data_loader
#         #     model.zero_grad()
#         #     # calculate the loss of the model on batched dataset through task-specified calculator
#         #     loss = self.calculator.compute_loss(model, batch_data)['loss']
#         #     loss.backward()
#         #     if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
#         #     optimizer.step()
#         # return
#         # for iter in range(self.num_steps):
#         #     for batch_data in data_loader:
#         #         inputs, labels = batch_data
#         #         model.zero_grad()
#         #         optimizer.zero_grad()
#         #         # 计算模型在批数据上的损失
#         #         loss = self.calculator.compute_loss(model, (inputs, labels))['loss']
#         #         loss.backward()
#         #         if self.clip_grad > 0:
#         #             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
#         #         optimizer.step()
#         # return
#         for iter in range(self.num_steps):
#             # get a batch of data
#             batch_data = self.get_batch_data(data_loader)
#             model.zero_grad()
#             optimizer.zero_grad()
#             # calculate the loss of the model on batched dataset through task-specified calculator
#             loss = self.calculator.compute_loss(model, batch_data)['loss']
#             loss.backward()
#             if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
#             optimizer.step()
#         return
# class fednova:
#     Server = Server
#     Client = Client



task = './my_task1'
# flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=10), task)
# runner = flgo.init(task, fednova, option={'num_rounds':20, 'local_test':True, 'learning_rate':0.005, })
# runner.run()
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fednova']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)