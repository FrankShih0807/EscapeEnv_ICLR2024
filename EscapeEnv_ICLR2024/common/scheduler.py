from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class ConstantParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value,last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        super(ConstantParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        return self.start_value

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value

class LinearParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value, end_value, total_steps, last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = int(total_steps)
        super(LinearParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        if self.last_epoch > self.total_steps:
            return self.end_value
        return max(self.start_value + (self.end_value - self.start_value) * (self.last_epoch / self.total_steps), self.end_value)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value

# class PolynomialParamScheduler(_LRScheduler):
#     def __init__(self, optimizer, param_name, start_value, end_value, total_steps, last_epoch=-1):
#         self.param_name = param_name
#         self.start_value = start_value
#         self.end_value = end_value
#         self.power = np.log(start_value / end_value) / np.log(total_steps)
#         self.total_steps = int(total_steps)
#         super(PolynomialParamScheduler, self).__init__(optimizer, last_epoch)

#     def get_param_value(self):
#         if self.last_epoch > self.total_steps:
#             return self.end_value
#         return max(self.start_value * (1/(self.last_epoch+1)) ** self.power, self.end_value)

#     def step(self, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch
#         self.param_value = self.get_param_value()
#         for param_group in self.optimizer.param_groups:
#             param_group[self.param_name] = self.param_value
            
class PolynomialParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value, power, last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        self.power = power
        super(PolynomialParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        steps = 10000
        return self.start_value * (steps/(self.last_epoch+steps)) ** self.power

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value
            
            
class CyclicalParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value, cycle_len=0, last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        self.cycle_len = cycle_len
        super(CyclicalParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        if self.cycle_len == 0:
            return self.start_value
        elif self.cycle_len > 0:
            return self.start_value/2 * (1 + np.cos(np.pi * np.mod(self.last_epoch, self.cycle_len)/self.cycle_len))

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value
            
            
class GeneralParamScheduler(_LRScheduler):
    def __init__(self, optimizer, param_name, start_value, cycle_len=0, last_epoch=-1):
        self.param_name = param_name
        self.start_value = start_value
        self.cycle_len = cycle_len
        super(GeneralParamScheduler, self).__init__(optimizer, last_epoch)

    def get_param_value(self):
        return self.start_value/2 * (1 + np.cos(np.pi * np.mod(self.last_epoch, self.cycle_len)/self.cycle_len))

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.param_value = self.get_param_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = self.param_value