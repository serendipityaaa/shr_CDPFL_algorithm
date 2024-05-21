import flgo.algorithm.fedbase
import copy
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np
from opacus import PrivacyEngine
import copy


class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'p':0, 's':1.0, 'eta':0.1})

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        super().initialize()
        self.last_valid_activations = None
        self.model = None
        self.weights = None
        self.start_phase = True
        self.num_pre_loss = 10
        self.threshold = 0.1

    def pack(self, model, *args, **kwargs):
        r"""
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.

        Args:
            model: the locally trained model

        Returns:
            package: a dict that contains the necessary information for the server
        """
        return {
            "model": copy.deepcopy(self.model)
        }

    def unpack(self, svr_pkg):
        global_model = svr_pkg['model']
        # initialize local model to the global model if local model is None,
        # and deactivate ALA at the 1st communication iteration by recoginizing the first round automatically
        if self.model is None:
            self.model = copy.deepcopy(global_model)
            return self.model

        # load the global encoder into local model
        params_global = list(global_model.parameters())
        params_local = list(self.model.parameters())
        for param, param_g in zip(params_local[:-self.p], params_global[:-self.p]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        local_model_temp = copy.deepcopy(self.model)
        params_local_temp = list(local_model_temp.parameters())

        # only consider higher layers
        params_local_head = params_local[-self.p:] # local model
        params_global_head = params_global[-self.p:] # global model
        params_local_temp_head = params_local_temp[-self.p:] # copy of local model

        # frozen the graident of the encoder in temp local model for efficiency
        for param in params_local_temp[:-self.p]: param.requires_grad = False

        # adaptively aggregate local model and global model by the weight into local temp model's head
        if self.weights is None: self.weights = [torch.ones_like(param.data).to(self.device) for param in params_local_head]
        for param_t, param, param_g, weight in zip(params_local_temp_head, params_local_head, params_global_head, self.weights):
            param_t.data = param + (param_g - param) * weight
        # weight learning
        # randomly sample partial local training data
        rand_num = int(self.s * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)
        rand_loader = DataLoader(Subset(self.train_data, list(range(rand_idx, rand_idx + rand_num))), self.batch_size, drop_last=True)
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        # train local aggregation weights (line 8-14)
        #

        try:
            for batch_data in rand_loader:
                loss = self.calculator.compute_loss(local_model_temp, batch_data)['loss']
                loss.backward()  # 正常反向传播
                self.last_valid_activations = copy.deepcopy(local_model_temp.activations)  # 假设activations是可访问的

                for param_t, param, param_g, weight in zip(params_local_temp_head, params_local_head,
                                                           params_global_head, self.weights):
                    weight.data = torch.clamp(weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)
                    param_t.data = param + (param_g - param) * weight
                losses.append(loss.item())
                cnt += 1
                if not self.start_phase: break
                if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                    self.gv.logger.info(
                        'Client:{}\tStd:{}\tALA epochs:{}'.format(self.id, np.std(losses[-self.num_pre_loss:]), cnt))
                    break
        except IndexError as e:
            if self.last_valid_activations is not None:
                print("Using last valid activations for backward pass")
                local_model_temp.activations = self.last_valid_activations
                for batch_data in rand_loader:
                    loss = self.calculator.compute_loss(local_model_temp, batch_data)['loss']
                    loss.backward()  # 使用上一次有效的激活重新尝试反向传播
            else:
                raise ValueError("No valid activations available for backward pass")

        self.start_phase = False

        # copy the aggregated head into local model (line 15)
        for param, param_t in zip(params_local_head, params_local_temp_head):
            param.data = param_t.data.clone()
        return self.model

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
        EPSILON = 50.0
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