import math
import os

import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from gaze_estimation_models_pytorch import GazeEstimationModelResnet18
from rtgene_dataset import RTGENEH5Dataset


class RTGENELearningRateFinder(object):

    def __init__(self, model, optimiser, loss, batch_size=128):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter()

        _root_dir = os.path.dirname(os.path.realpath(__file__))

        data_train = RTGENEH5Dataset(h5_file=h5py.File(os.path.abspath(os.path.join(_root_dir, "../../../RT_GENE/dataset.hdf5")), mode="r"),
                                     subject_list=list(range(16)))

        dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)

        # Train and evaluate
        logs, losses = self.find_lr(model=model, dataloader=dataloader, criterion=loss, optimiser=optimiser, batch_size=batch_size,
                                    epoch_length=len(data_train))
        plt.plot(logs[10:-5], losses[10:-5])
        plt.xscale('log')
        plt.show()

    def find_lr(self,  dataloader, model, optimiser, criterion, init_value=1e-6, final_value=1e-3, beta=0.98, epoch_length=100000, batch_size=64):
        num = (epoch_length // batch_size) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        optimiser.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []

        additional_steps = epoch_length // batch_size
        _rtgene_model = model.to(self._device)
        model.eval()

        data_iter = iter(dataloader)

        # Start training
        with trange(0, additional_steps) as pbar:
            for step in pbar:
                batch_num += 1
                # As before, get the loss for this mini-batch of inputs/outputs
                try:
                    batch = next(data_iter)
                except StopIteration:
                    return log_lrs, losses

                _left_patch, _right_patch, _labels, _head_pose = batch

                _left_patch = _left_patch.to(self._device)
                _right_patch = _right_patch.to(self._device)
                _labels = _labels.to(self._device).float()
                _head_pose = _head_pose.to(self._device).float()

                optimiser.zero_grad()
                with torch.set_grad_enabled(True):

                    # Get model outputs and calculate loss
                    angular_out = _rtgene_model(_left_patch, _right_patch, _head_pose)
                    loss = criterion(angular_out, _labels)

                    # Compute the smoothed loss
                    avg_loss = beta * avg_loss + (1 - beta) * loss.item()
                    smoothed_loss = avg_loss / (1 - beta ** batch_num)
                    # Stop if the loss is exploding
                    # if batch_num > 1 and smoothed_loss > 4 * best_loss:
                    #    return log_lrs, losses
                    # Record the best loss
                    if smoothed_loss < best_loss or batch_num == 1:
                        best_loss = smoothed_loss
                    # Store the values
                    losses.append(smoothed_loss)
                    log_lrs.append(lr)
                    # Do the SGD step
                    loss.backward()
                    optimiser.step()
                    # Update the lr for the next step
                    lr *= mult
                    optimiser.param_groups[0]['lr'] = lr

                    pbar.set_description("Learning Rate: {:4.8e}, Loss: {:4.8f}".format(lr, smoothed_loss))
                    pbar.update()

                    self.writer.add_scalar("data/lr", math.log10(lr), global_step=batch_num)
                    self.writer.add_scalar("data/loss", smoothed_loss, global_step=batch_num)

        return log_lrs, losses


if __name__ == "__main__":
    rt_gene_fast_model = GazeEstimationModelResnet18(num_out=2)
    params_to_update = []
    for name, param in rt_gene_fast_model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    learning_rate = 1e-1
    optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, betas=(0.9, 0.95))
    criterion = torch.nn.MSELoss(reduction="sum")

    RTGENELearningRateFinder(model=rt_gene_fast_model, optimiser=optimizer, loss=criterion, batch_size=128)
