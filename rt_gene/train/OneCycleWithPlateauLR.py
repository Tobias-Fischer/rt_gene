import torch


class OneCycleWithPlateauLR(object):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 num_steps: int,
                 cutoffs: tuple = (0.0, 0.01, 0.8, 0.85),
                 lr_range: tuple = (0.1, 1.),
                 momentum_range: tuple = (0.85, 0.95),
                 annihilation_frac: float = 0.1,
                 reduce_factor: float = 0.01,
                 last_step: int = -1):
        # Sanity check
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.num_steps = num_steps

        self.min_lr, self.max_lr = lr_range[0], lr_range[1]
        self.current_lr = 0.0
        assert self.min_lr < self.max_lr, \
            "Argument lr_range must be (min_lr, max_lr), where min_lr < max_lr"

        self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
        self.current_momentum = 1.0
        assert self.min_momentum < self.max_momentum, \
            "Argument momentum_range must be (min_momentum, max_momentum), where min_momentum < max_momentum"

        assert len(cutoffs) == 4, "Length of Cuttoffs must be 4: start of increase (usually 0), plateau, start of decrease and then annihilation"
        self.cutoffs = cutoffs

        self.final_lr = self.min_lr * reduce_factor

        self.last_step = last_step

        if self.last_step == -1:
            self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_momentum(self):
        return self.optimizer.param_groups[0]['momentum']

    def step(self):
        """Conducts one step of learning rate and momentum update
        """
        current_step = self.last_step + 1
        self.last_step = current_step

        if current_step >= self.num_steps * self.cutoffs[3]:
            # annihilation
            scale = (current_step - self.num_steps * self.cutoffs[3]) / (self.num_steps - self.num_steps * self.cutoffs[3])
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None
        elif current_step >= self.num_steps * self.cutoffs[2]:
            # Scale down phase
            scale = (current_step - self.num_steps * self.cutoffs[2]) / (self.num_steps - self.num_steps * self.cutoffs[3])
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        elif current_step >= self.num_steps * self.cutoffs[1]:
            # plateau phase
            lr = self.max_lr
            momentum = self.min_momentum
        elif current_step >= self.num_steps * self.cutoffs[0]:
            # Scale up phase
            scale = current_step / (self.num_steps * self.cutoffs[1])
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        else:
            # if we're here it means we're probably retraining on a higher epoch rate; therefore just use the annihilation rate
            scale = ((self.num_steps - 1) - self.num_steps * self.cutoffs[3]) / (self.num_steps - self.num_steps * self.cutoffs[3])
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None

        self.current_lr = lr
        self.current_momentum = momentum

        self.optimizer.param_groups[0]['lr'] = lr
        if momentum:
            self.optimizer.param_groups[0]['momentum'] = momentum
