import flgo.algorithm.fedbase as fedbase
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fmodule
import os
import flgo
import copy
import torch
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import flgo.experiment.analyzer
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

class Server(fedbase.BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu':0.01})
class Client(fedbase.BasicClient):
    def initialize(self):
        super().initialize()  # 确保也调用了基类的初始化方法
        self.model = None
        self.privacy_engine_initialized = False
    @fmodule.with_multi_gpus
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
    def train(self, model):
        # 记录全局模型参数\global parameters
        src_model = copy.deepcopy(model)
        # 冻结全局模型梯度
        src_model.freeze_grad()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        model.train()
        actual_model = copy.deepcopy(model)
        data_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.loader_num_workers,
            pin_memory=self.option['pin_memory'],
            drop_last=not self.option.get('no_drop_last', False)
        )
        if not hasattr(data_loader, 'get_device'):  # 检查data_loader是否有get_device方法
            data_loader = getattr(data_loader, 'data_loader', data_loader)  # 如果没有，则可能是一个包装对象，尝试获取其内部的真实DataLoader对象

        if not self.privacy_engine_initialized:
            MAX_GRAD_NORM = 1.2
            EPSILON = 50.0
            DELTA = 1e-5
            EPOCHS = 5
            privacy_engine = PrivacyEngine()
            try:
                model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=data_loader,
                    epochs=EPOCHS,
                    target_epsilon=EPSILON,
                    target_delta=DELTA,
                    max_grad_norm=MAX_GRAD_NORM,
                )
            except Exception as e:
                print("Error during making the model private:", e)
                raise
            self.privacy_engine_initialized = True
        # actual_model = copy.deepcopy(model)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data(data_loader)
            model.zero_grad()
            actual_model.zero_grad()
            optimizer.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            # 计算近端项损失
            loss_proximal = 0
            for pm, ps in zip(actual_model.parameters(), src_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            loss = loss + 0.5 * self.mu * loss_proximal
            loss.backward()
            optimizer.step()
        return
class fedprox:
    Server = Server
    Client = Client
# task = './test_synthetic'
# config = {'benchmark':{'name':'flgo.benchmark.synthetic_regression', 'para':{'alpha':0.5, 'beta':0.5, 'num_clients':30}}}
# if not os.path.exists(task): flgo.gen_task(config, task_path = task)
# option = {'num_rounds':30, 'num_epochs':1, 'batch_size':8, 'learning_rate':0.1}
# fedavg_runner = flgo.init(task, fedavg, option=option)
# fedprox_runner = flgo.init(task, fedprox, option=option)
# fedavg_runner.run()
# fedprox_runner.run()
# import flgo.experiment.analyzer
# analysis_plan = {
#     'Selector':{
#         'task': task,
#         'header':['fedavg', 'fedprox']
#     },
#     'Painter':{
#         'Curve':[
#             {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on Synthetic'}},
#             {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on Synthetic'}},
#         ]
#     }
# }
# flgo.experiment.analyzer.show(analysis_plan)
task = './my_task3'
flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=10), task)
runner = flgo.init(task, fedprox, option={'num_rounds':20, 'local_test':True, 'learning_rate':0.005, })
runner.run()
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['pfedme']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)
