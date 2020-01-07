import os
import torch
import numpy as np
from torch.utils.data import DataLoader


class ModelUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def make_train_state(args):
        state_dict = dict(
            stop_early=False,
            early_stopping_step=0,
            early_stopping_best_val=1e8,
            learing_rate=args.learning_rate,
            epoch_index=0,
            train_loss=[],
            train_acc=[],
            val_loss=[],
            val_acc=[],
            test_loss=-1,
            test_acc=-1,
            model_filename=args.model_state_file
        )
        return state_dict

    @staticmethod
    def update_train_state(args, model, train_state):

        # Save model for the very first time
        if train_state['epoch_index'] == 0:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['stop_early'] = False

        elif train_state["epoch_index"] >= 1:
            loss_tm1, loss_t = train_state['val_loss'][-2:]
            # if loss is worse
            if loss_t >= train_state['early_stopping_best_val']:
                train_state['early_stopping_step'] += 1
            # if loss is better
            else:
                if loss_t < train_state['early_stopping_best_val']:
                    torch.save(model.state_dict(), train_state['model_filename'])
                    # reset early stopping step
                    train_state['early_stopping_step'] = 0
            train_state['stop_early'] = train_state['early_stopping_step'] >= args.early_stopping_criteria
        return train_state

    @staticmethod
    def compute_accuracy(y_pred, y_target):
        # It is a vector operation, not a single operation
        _, y_pred_indices = y_pred.max(dim=1)
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    @staticmethod
    def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
        # Implement torch Dataloader function for this
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle, drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

    @staticmethod
    def set_seed_everywhere(seed, cuda):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def handle_dirs(save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
