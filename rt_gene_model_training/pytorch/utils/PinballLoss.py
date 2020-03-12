import torch


class PinballLoss(object):

    def __init__(self, reduction="mean"):
        super(PinballLoss, self).__init__()
        self.q1 = 0.45
        self.q9 = 1 - self.q1

        _reduction_strategies = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x
        }
        assert reduction in _reduction_strategies.keys(), "Reduction method unknown, possibilities include 'mean', 'sum' and 'none'"

        self._reduction_strategy = _reduction_strategies.get(reduction)

    def __call__(self, output, target):
        angle_o = output[:, :2]
        var_o = output[:, 2:3]
        var_o = var_o.view(-1, 1).expand(var_o.size(0), 2)

        q_10 = target - (angle_o - var_o)
        q_90 = target - (angle_o + var_o)

        loss_10 = torch.max(self.q1 * q_10, (self.q1 - 1) * q_10)
        loss_90 = torch.max(self.q9 * q_90, (self.q9 - 1) * q_90)

        loss_10 = self._reduction_strategy(loss_10)
        loss_90 = self._reduction_strategy(loss_90)

        return loss_10 + loss_90
