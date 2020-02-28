import numpy as np
import torch


class PinballLoss(object):

    def __init__(self):
        super(PinballLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1 - self.q1

    def __call__(self, output, target):
        output_o = output[:, :2]
        var_o = output[:, 2:3]
        var_o = var_o.view(-1, 1).expand(var_o.size(0), 2)

        q_10 = target - (output_o - var_o)
        q_90 = target - (output_o + var_o)

        loss_10 = torch.max(self.q1 * q_10, (self.q1 - 1) * q_10)
        loss_90 = torch.max(self.q9 * q_90, (self.q9 - 1) * q_90)

        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10 + loss_90


class PinballMeter(object):
    def __init__(self, batch_size):
        super(PinballMeter, self).__init__()
        self.__pb = PinballLoss()
        self.reset()
        self.val = 0
        self.batch_size = batch_size

    def add(self, output, target):
        value = self.__pb(output=output, target=target)
        value = value.detach().cpu().numpy()
        self.val = value
        self.sum += value * self.batch_size

        self.mean = self.mean_old + self.batch_size * (value - self.mean_old) / float(self.n + self.batch_size)
        self.m_s += self.batch_size * (value - self.mean_old) * (value - self.mean)
        self.mean_old = self.mean
        self.std = np.sqrt(self.m_s / (self.n + self.batch_size - 1.0))
        self.var = self.std ** 2

        self.n += self.batch_size

    def value(self):
        return float(self.mean)

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


if __name__ == "__main__":
    _target = torch.from_numpy(np.array([[0.5, 0.5], [0.1, 0.1]]))
    _output = torch.from_numpy(np.array([[0.5, 0.5, 0.1], [0.2, 0.2, 0.2]]))

    pbl = PinballLoss()
    meter = PinballMeter(batch_size=2)
    loss = pbl(_output, _target)
    print(loss)
    meter.add(_output, _target)
    print(meter.value())

