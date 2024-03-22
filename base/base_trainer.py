from abc import abstractmethod

import torch
from numpy import inf


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, optimizer, config, writer=None, rank=0, ddp=False, epoch_inter=1):
        self.config = config
        self.rank = rank
        if rank == 0:
            self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        else:
            self.logger = None

        self.epoch_inter = epoch_inter

        # setup GPU device if available, move models into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        if not ddp:
            self.model = model.to(self.device)
        else:
            self.model = model
        # if len(device_ids) > 1:
        #     self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        if type(optimizer) == list:
            self.optimizer = optimizer[0]
            self.var_optimizer = optimizer[1]
        else:
            self.optimizer = optimizer
            self.var_optimizer = None

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.best_results = {}

        # configuration to monitor models performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = writer

        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if self.rank == 0 and epoch % self.epoch_inter == 0:
                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(result)

                # print logged informations to the screen
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

                # evaluate models performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off' and self.rank == 0:
                    try:
                        # check whether models performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    # Do not save nan results
                    if self.mnt_mode == 'min' and log[self.mnt_metric] == 0:
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                        self.best_results = log.copy()
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.early_stop))
                        break

                if epoch % self.save_period == 0 and self.rank == 0:
                    if best is True:
                        self._save_checkpoint(epoch, save_best=True)
                    self._save_checkpoint(epoch, save_last=True)

        # end
        if self.rank == 0:
            print('Best Results:')
            for k in self.best_results:
                print(k, self.best_results[k])


    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move models into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False, save_last=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if self.var_optimizer is not None:
            state['var_optimizer'] = self.var_optimizer.state_dict()
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
        elif save_last:
            last_path = str(self.checkpoint_dir / 'model_last.pth')
            torch.save(state, last_path)
            self.logger.info("Saving lastest: model_last.pth ...")
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        if self.rank == 0:
            self.logger.info("Loading checkpoint: {} ...".format(resume_path))
            print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            k_ = k[7:] if k.startswith("module.") else k
            state_dict[k_] = v
        self.model.load_state_dict(state_dict)

        # load opt
        if self.rank == 0:
            print("Loading optimizer: {} ...".format(resume_path))
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
