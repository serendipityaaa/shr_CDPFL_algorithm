"""
This is a non-official implementation of personalized FL method pFedMe (http://arxiv.org/abs/2006.08848).
The original implementation is in github repo (https://github.com/CharlieDinh/pFedMe/)
"""
import copy
import torch
import flgo.utils.fmodule as fmodule
import flgo.algorithm.fedbase
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'beta':1.0, 'lambda_reg':15.0, 'K':5})

    def aggregate(self, models):
        return (1-self.beta)*self.model+self.beta*fmodule._model_average(models)

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = None
        self.privacy_engine_initialized = False
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
        if self.model is None:
            self.model = copy.deepcopy(model)
        self.model.train()  # 确保模型是在训练模式
        optimizer = self.calculator.get_optimizer(self.model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        data_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.loader_num_workers,
            pin_memory=self.option['pin_memory'],
            drop_last=not self.option.get('no_drop_last', False)
        )
        if not self.privacy_engine_initialized:
            MAX_GRAD_NORM = 1.2
            EPSILON = 50.0
            DELTA = 1e-5
            EPOCHS = 5
            privacy_engine = PrivacyEngine()
            self.model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=data_loader,
                epochs=EPOCHS,
                target_epsilon=EPSILON,
                target_delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
            )
            self.privacy_engine_initialized = True
        # for iter in range(self.num_steps):
        #     # line 7 in Algorithm 1 pFedMe
        #     batch_data = self.get_batch_data(data_loader)
        #     self.model.load_state_dict(model.state_dict())
        #     self.model.train()
        #     self.model.zero_grad()
        #     model.freeze_grad()
        #     for _ in range(self.K):
        #         loss = self.calculator.compute_loss(self.model, batch_data)['loss']
        #         for param_theta_i, param_wi in zip(self.model.parameters(), model.parameters()):
        #             loss += self.lambda_reg*0.5*torch.sum((param_theta_i-param_wi)**2)
        #         loss.backward()
        #         if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.clip_grad)
        #         optimizer.step()
        #     model.zero_grad()
        #     optimizer.zero_grad()
        #     model.enable_grad()
        #     # line 8
        #     with torch.no_grad():
        #         for param_wi, param_thetai in zip(model.parameters(), self.model.parameters()):
        #             param_wi.data = param_wi.data - self.learning_rate * self.lambda_reg * (param_wi.data - param_thetai)
        #     self.model.load_state_dict(model.state_dict())
        for _ in range(self.num_steps):
            batch_data = self.get_batch_data(data_loader)

            # 解决Opacus模型封装导致的问题: 直接访问内部模型
            # Added handling to access the internal model if it is wrapped by Opacus's GradSampleModule
            actual_model = self.model._module if hasattr(self.model, '_module') else self.model
            actual_model.train()
            actual_model.zero_grad()
            model.freeze_grad()

            for _ in range(self.K):
                loss = self.calculator.compute_loss(actual_model, batch_data)['loss']
                for param_theta_i, param_wi in zip(actual_model.parameters(), model.parameters()):
                    loss += self.lambda_reg * 0.5 * torch.sum((param_theta_i - param_wi) ** 2)
                loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(actual_model.parameters(), self.clip_grad)
                optimizer.step()
                actual_model.zero_grad()
                optimizer.zero_grad()

            model.zero_grad()
            optimizer.zero_grad()
            model.enable_grad()

            with torch.no_grad():
                for param_wi, param_thetai in zip(model.parameters(), actual_model.parameters()):
                    param_wi.data -= self.learning_rate * self.lambda_reg * (param_wi.data - param_thetai.data)

        return
