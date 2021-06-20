import os
import datetime
import shutil
import logging
import random
import numpy as np
import math
from einops import rearrange

from utils import Cutout

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


def setup_logger(name, file_path, level=logging.INFO):
    formatter = logging.Formatter(datefmt='[%Y-%m-%d_%H:%M:%S]', fmt='%(asctime)s %(message)s')
    handler = logging.FileHandler(file_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class Trainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

        self._init_seed()
        self._init_dataloaders()
        self._init_device()
        self._init_dirs()
        self._init_loggers()
        self._init_writer()
        self._init_print_basic_info()

    def _init_seed(self):
        seed = int(self.cfg.random_seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def _init_dataloaders(self):
        dict_datasets = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}
        dataset = dict_datasets[self.cfg.dataset.lower()]
        # Image Preprocessing
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])

        if self.cfg.cutout:
            train_transform.transforms.append(Cutout(n_holes=self.cfg.n_holes, length=self.cfg.length))

        if self.cfg.is_validation: # only perform validation
            self.train_set = None
            self.val_set = dataset(root='data/', train=False, transform=val_transform, download=False)
        elif self.cfg.epoch_validation == 0: # training without validation
            self.train_set = dataset(root='data/', train=True, transform=train_transform, download=False)
            self.val_set = None
        else: # training with validation
            self.train_set = dataset(root='data/', train=True, transform=train_transform, download=False)
            self.val_set = dataset(root='data/', train=False, transform=val_transform, download=False)

        if self.train_set is not None:
            self.train_loader = DataLoader(
                dataset=self.train_set,
                batch_size=self.cfg.batch_size_training,
                num_workers=self.cfg.num_workers_training,
            )
        else:
            self.train_loader = None
        if self.val_set is not None:
            self.val_loader = DataLoader(
                dataset=self.val_set,
                batch_size=self.cfg.batch_size_validation,
                num_workers=self.cfg.num_workers_validation,
            )
        else:
            self.val_loader = None

    def _init_device(self):
        idx_dev = self.cfg.index_device
        if idx_dev >= 0:
            name_device = f'cuda:{idx_dev}'
        elif idx_dev == -1:
            name_device = 'cpu'
        else:
            raise ValueError(f"Unknown device index: {idx_dev}")
        self.device = torch.device(name_device)

    def _init_dirs(self):
        self.time_experiment = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.name_experiment = '_'.join(['exp', self.time_experiment, self.cfg.name_experiment])
        self.path_experiment = os.path.join(self.cfg.path_saving, self.name_experiment)
        self.path_checkpoints = os.path.join(self.path_experiment, 'checkpoints')
        self.path_visualization = os.path.join(self.cfg.path_saving, 'runs', self.name_experiment)
        self.path_log = os.path.join(self.path_experiment, 'logs')
        os.makedirs(self.path_log)
        os.makedirs(self.path_checkpoints)
        os.makedirs(self.path_visualization)

        if self.cfg.delete_previous_results and os.path.exists(self.cfg.path_saving):
            list_dirs_exps = os.listdir(self.cfg.path_saving)
            for dir in list_dirs_exps:
                if dir != self.name_experiment and dir[:3] == 'exp':
                    shutil.rmtree(os.path.join(self.cfg.path_saving, dir))
                    shutil.rmtree(os.path.join(self.cfg.path_saving, 'runs', dir))

    def _init_loggers(self):
        self.logger_all = setup_logger('all', os.path.join(self.path_log, 'output.txt'))
        self.logger_training = setup_logger('train', os.path.join(self.path_log, 'training.txt'))
        self.logger_validation = setup_logger('val', os.path.join(self.path_log, 'validation.txt'))

    def _init_writer(self):
        self.writer = SummaryWriter(self.path_visualization)

    def _init_print_basic_info(self):
        self.logger_all.info(f'Dir for this experiment: {self.path_experiment}')
        self.logger_all.info(self.cfg)
        self.logger_all.info(self.model)

    def _get_optimizer(self, params):
        name_opt = self.cfg.name_optimizer.lower()

        if name_opt == 'sgd':
            optimizer = optim.SGD(
                params=params,
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
                momentum=self.cfg.momentum,
                nesterov=not self.cfg.no_nesterov,
            )
        elif name_opt in ('adam', 'adamw'):
            optimizer = optim.SGD(
                params=params,
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise ValueError(f'Unknown optimizer: {name_opt}')

        name_sch = self.cfg.name_scheduler.lower()
        if name_sch is not None:
            if name_sch == 'cycliclr':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                        base_lr=self.cfg.scheduler_cycliclr_base_learning_rate,
                                                        max_lr=self.cfg.scheduler_cycliclr_max_learning_rate,
                                                        mode=self.cfg.scheduler_cycliclr_mode,
                                                        gamma=self.cfg.scheduler_cycliclr_gamma,
                                                        cycle_momentum=self.cfg.scheduler_cycliclr_cycle_momentum)
            elif name_sch == 'multisteplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.cfg.scheduler_multisteplr_milestones,
                    gamma=self.cfg.scheduler_multisteplr_gamma,
                )
            else:
                raise ValueError(f'Unknown scheduler: {name_sch}')
        else:
            scheduler = None

        return optimizer, scheduler

    def train(self):
        num_batches_train = len(self.train_set) // self.cfg.batch_size_training

        self.model = self.model.to(self.device)
        optimizer, scheduler = self._get_optimizer(self.model.parameters())

        self.logger_all.warning(f'------ Training: {num_batches_train} iters/batches per epoch ------')
        score_model_best = -math.inf
        names_losses = self.model.names_losses
        for epoch in range(self.cfg.num_epochs):
            base_iter = epoch * len(self.train_loader)
            self.model.train()
            average_losses = 0.
            self.model.num_pics = 0.
            self.model.num_correct_pred = 0.
            for iter, data in enumerate(self.train_loader):
                total_iter = base_iter + iter

                optimizer.zero_grad()
                input, ground_truth = data
                input, ground_truth = input.to(self.device), ground_truth.to(self.device)

                input, ground_truth = self.model.preprocessing(input, ground_truth, self.device)

                output = self.model(input)
                losses, total_loss = self.model.total_loss(output, ground_truth)
                total_loss.backward()
                optimizer.step()

                for idx in range(len(names_losses)):
                    self.writer.add_scalar(f'training/{names_losses[idx]}', losses[idx] / output.size(0), total_iter)
                average_losses += losses * output.size(0)

            if not self.cfg.mixup and not self.cfg.cutmix:
                self.writer.add_scalar(f'training/accuracy', self.model.num_correct_pred / self.model.num_pics, epoch)

            # validation
            if self.val_set is not None and epoch % self.cfg.epoch_validation == 0:
                # save current model
                if self.cfg.save_all_models:
                    name_model_saved = f'model_{epoch}_{iter}.pth'
                    self.logger_all.info('Saving current model: ' + name_model_saved)
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, name_model_saved))

                average_metrics, score = self.validation()

                # visualization and log for validation
                names_metrics = self.model.names_metrics_val
                for idx in range(len(average_metrics)):
                    self.writer.add_scalar(f'validation/{names_metrics[idx]}', average_metrics[idx], epoch)
                self.writer.add_scalar('validation/score_model', score, epoch)
                result_log = [f'epoch: {epoch}']
                for idx in range(len(average_metrics)):
                    result_log.append(f'{names_metrics[idx]}: {average_metrics[idx]:.4f}')
                result_log.append(f'score: {score:.4f}')
                info_logged = ', '.join(result_log)
                self.logger_all.info(f'[valid] {info_logged}')
                self.logger_validation.info(info_logged)

                # save best model
                if score >= score_model_best:
                    if self.cfg.save_better_models:
                        name_best_model_saved = f'model_best_{epoch}_{iter}.pth'
                    else:
                        name_best_model_saved = 'model_best.pth'
                    self.logger_all.warning('Saving best model: ' + name_best_model_saved)
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, name_best_model_saved))

                self.model.train()

            if scheduler:
                scheduler.step()

            # visualization and log for training
            average_losses /= len(self.train_set)
            result_log = [f'epoch: {epoch}']
            for idx in range(len(average_losses)):
                result_log.append(f'{names_losses[idx]}: {average_losses[idx]:.4f}')
            info_logged = ', '.join(result_log)
            self.logger_all.info(f'[train] {info_logged}')
            self.logger_training.info(info_logged)

        self.logger_all.info(f'training finished')

    def validation(self):
        self.model.eval()

        with torch.no_grad():
            average_metrics = 0.
            for _, data in enumerate(self.val_loader):
                input, ground_truth = data
                input, ground_truth = input.to(self.device), ground_truth.to(self.device)
                output = self.model.forward_val(input)
                metrics = self.model.metrics_val(output, ground_truth)
                average_metrics += metrics

            score = self.model.score_val(average_metrics)

        return torch.tensor([average_metrics[0] / average_metrics[1]]), score
