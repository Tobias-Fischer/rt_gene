from evaluate_blink_model import threefold_evaluation
from train_blink_model import ThreefoldTraining
from dataset_manager import RTBeneDataset
from pathlib import Path
import argparse
import pprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_save_root", help="target folder to save the models (auto-saved)")
    parser.add_argument("csv_subject_list", help="path to the dataset csv file")
    parser.add_argument("--ensemble_size", type=int, default=1, help="number of models to train for the ensemble")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--input_size", type=tuple, help="input size of images", default=(96, 96))

    args = parser.parse_args()

    fold_list = ['fold1', 'fold2', 'fold3']
    ensemble_size = args.ensemble_size  # 1 is considered as single model
    epochs = args.epochs
    batch_size = args.batch_size
    input_size = args.input_size
    csv_subject_list = args.csv_subject_list
    model_save_root = args.model_save_root

    dataset = RTBeneDataset(csv_subject_list, input_size)

    threefold_training = ThreefoldTraining(dataset, epochs, batch_size, input_size)

    all_evaluations = {}

    for backbone in ['densenet121', 'resnet50', 'mobilenetv2']:
        models_fold1 = []
        models_fold2 = []
        models_fold3 = []

        for i in range(1, ensemble_size + 1):
            model_save_path = Path(model_save_root + backbone + '/' + str(i))
            model_save_path.mkdir(parents=True, exist_ok=True)
            threefold_training.train(backbone, str(model_save_path) + '/')

            models_fold1.append(str(model_save_path) + '/rt-bene_' + backbone + '_fold1_best.h5')
            models_fold2.append(str(model_save_path) + '/rt-bene_' + backbone + '_fold2_best.h5')
            models_fold3.append(str(model_save_path) + '/rt-bene_' + backbone + '_fold3_best.h5')

        evaluation = threefold_evaluation(dataset, models_fold1, models_fold2, models_fold3, input_size)
        all_evaluations[backbone] = evaluation

    threefold_training.free()

    pprint.pprint(all_evaluations)
