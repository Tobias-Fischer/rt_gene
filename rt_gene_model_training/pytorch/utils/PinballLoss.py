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

