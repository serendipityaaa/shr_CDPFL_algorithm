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
        self.init_algo_para({'mu':0.1})
        self.sample_option = 'md'
        self.aggregation_option = 'uniform'

class Client(fedbase.BasicClient):
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
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        data_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.loader_num_workers,
            pin_memory=self.option['pin_memory'],
            drop_last=not self.option.get('no_drop_last', False)
        )
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        MAX_GRAD_NORM = 1.2
        EPSILON = 100.0
        DELTA = 1e-5
        EPOCHS = 5
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            # noise_multiplier=1.0,
            # max_grad_norm=1.0,
            epochs=EPOCHS,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data(data_loader)
            model.zero_grad()
            optimizer.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss_proximal = 0
            for pm, ps in zip(model.parameters(), src_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            loss = loss + 0.5 * self.mu * loss_proximal
            loss.backward()
            optimizer.step()
        return

class fedprox:
    Server = Server
    Client = Client


task = './my_task3'
flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=10), task)
runner = flgo.init(task, fedprox, option={'num_rounds':20, 'local_test':True, 'learning_rate':0.005})
runner.run()
analysis_plan = {
    'Selector':{
        'task': task,
        'header':['fedprox']
    },
    'Painter':{
        'Curve':[
            {'args':{'x': 'communication_round', 'y':'val_loss'}, 'fig_option':{'title':'valid loss on MNIST'}},
            {'args':{'x': 'communication_round', 'y':'val_accuracy'},  'fig_option':{'title':'valid accuracy on MNIST'}},
        ]
    }
}
flgo.experiment.analyzer.show(analysis_plan)