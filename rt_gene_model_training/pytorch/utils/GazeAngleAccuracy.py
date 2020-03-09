import numpy as np


class GazeAngleAccuracy(object):

    def __call__(self, batch_y_pred, batch_y_true):
        batch = batch_y_true.size()[0]
        batch_y_pred = batch_y_pred.cpu().detach().numpy()
        batch_y_true = batch_y_true.cpu().detach().numpy()
        acc = []
        for i in range(batch):
            y_true, y_pred = batch_y_true[i], batch_y_pred[i]
            pred_x = -1 * np.cos(y_pred[0]) * np.sin(y_pred[1])
            pred_y = -1 * np.sin(y_pred[0])
            pred_z = -1 * np.cos(y_pred[0]) * np.cos(y_pred[1])
            pred = np.array([pred_x, pred_y, pred_z])
            pred = pred / np.linalg.norm(pred)

            true_x = -1 * np.cos(y_true[0]) * np.sin(y_true[1])
            true_y = -1 * np.sin(y_true[0])
            true_z = -1 * np.cos(y_true[0]) * np.cos(y_true[1])
            gt = np.array([true_x, true_y, true_z])
            gt = gt / np.linalg.norm(gt)

            acc.append(np.rad2deg(np.arccos(np.dot(pred, gt))))

        acc = np.mean(np.array(acc))
        return acc
