import numpy as np
import argparse
import pickle
import torch
import random
from evaluation import get_prediction_for_all_events,get_prediction_for_last_events
from sklearn.metrics import f1_score
import torch.utils.data as data_utils

from pathlib import Path
from sklearn.model_selection import train_test_split

from data import Sequence, SequenceDataset
from model import LogNormMix

from copy import deepcopy

seed = 0

# Model config
context_size = 64  # Size of the RNN hidden vector
mark_embedding_size = 32  # Size of the mark embedding (used as RNN input)
num_mix_components = 64  # Number of components for a mixture model
rnn_type = "GRU"  # What RNN to use as an encoder {"RNN", "GRU", "LSTM"}

# Training config
batch_size = 64  # Number of sequences in a batch
regularization = 1e-5  # L2 regularization parameter
learning_rate = 1e-4  # Learning rate for Adam optimizer
max_epochs = 5  # For how many epochs to train
display_step = 1  # Display training statistics after every display_step
patience = 50  # After how many consecutive epochs without improvement of val loss to stop training

TYPE_SIZE_DICT = {'retweet': 3, 'bookorder': 8, 'meme': 5000, 'mimic': 75, 'stackoverflow': 22,
                  'simulated': 2}

parser = argparse.ArgumentParser(description="Train the models.")
parser.add_argument('-e', '--epochs', type=int, default=1000,
                    help='number of epochs.')
parser.add_argument('-b', '--batch', type=int,
                    dest='batch_size', default=batch_size,
                    help='batch size. (default: {})'.format(batch_size))
parser.add_argument('--lr', default=learning_rate, type=float,
                    help="set the optimizer learning rate. (default {})".format(learning_rate))
parser.add_argument('--hidden', type=int,
                    dest='hidden_size', default=context_size,
                    help='number of hidden units. (default: {})'.format(context_size))
parser.add_argument('--mark_embedding', type=int,
                    dest='hidden_size', default=mark_embedding_size,
                    help='number of hidden units. (default: {})'.format(mark_embedding_size))
parser.add_argument('-mix_componenets', '--mix_componenets', type=int, default=num_mix_components,
                    help='mix componenets')
parser.add_argument('--lambda-l2', type=float, default=regularization,
                    help='regularization loss.')
parser.add_argument('--dev-ratio', type=float, default=0.1,
                    help='override the size of the dev dataset.')
parser.add_argument('--task', type=str, default='simulated',
                    help='task type')

parser.add_argument('-seed', '--seed', type=int, default=42,
                    help='seed')

args = parser.parse_args()
print(args)

# ------Reproducibility-------

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = 'cuda'
    # torch.set_default_tensor_type(cuda_tensor)
    torch.cuda.manual_seed(seed=args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    # torch.set_default_tensor_type(cpu_tensor)
    device = 'cpu'


# ------Reproducibility-------

def get_inter_times(seq: dict):
    """Get inter-event times from a sequence."""
    return np.ediff1d(np.concatenate([[seq["t_start"]], seq["arrival_times"], [seq["t_end"]]]))


dataset_name = args.task
num_marks = TYPE_SIZE_DICT[args.task]
dataset_path = 'data/' + dataset_name + '/'  # run dpp.data.list_datasets() to see the list of available datasets

with open(dataset_path + 'train.pkl', 'rb') as f:
    train = pickle.load(f)

with open(dataset_path + 'valid.pkl', 'rb') as f:
    valid = pickle.load(f)
with open(dataset_path + 'test.pkl', 'rb') as f:
    test = pickle.load(f)


def create_seq_data_set(dataset, num_marks):
    sequences = [
        Sequence(
            inter_times=get_inter_times(seq),
            marks=seq.get("marks"),
            t_start=seq.get("t_start"),
            t_end=seq.get("t_end")
        ).to(device)
        for seq in dataset["sequences"]
    ]
    dataset = SequenceDataset(sequences=sequences, num_marks=num_marks)

    return dataset


d_train = create_seq_data_set(train, num_marks)
d_val = create_seq_data_set(valid, num_marks)
d_test = create_seq_data_set(test, num_marks)

dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=False)
dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)

# Define the model
print('Building model...')
mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()

model = LogNormMix(
    num_marks=d_train.num_marks,
    mean_log_inter_time=mean_log_inter_time,
    std_log_inter_time=std_log_inter_time,
    context_size=context_size,
    mark_embedding_size=mark_embedding_size,
    rnn_type=rnn_type,
    num_mix_components=num_mix_components,
)
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)
# Traning
print('Starting training...')


def aggregate_loss_over_dataloader(dl):
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dl:
            total_loss += -model.log_prob(batch).sum()
            total_count += batch.mask.sum().item()
    return total_loss / total_count


impatient = 0
best_loss = np.inf
best_model = deepcopy(model.state_dict())
training_val_losses = []
training_events = d_train.total_num_events

for epoch in range(max_epochs):
    epoch_train_loss = 0
    model.train()
    for batch in dl_train:
        opt.zero_grad()
        # loss = -model.log_prob(batch)
        loss = -model.log_prob(batch).sum()
        loss.backward()
        epoch_train_loss += loss.detach()

        opt.step()

    model.eval()
    with torch.no_grad():
        loss_val = aggregate_loss_over_dataloader(dl_val)
        loss_test = aggregate_loss_over_dataloader(dl_test)

        training_val_losses.append(loss_val)

    if (best_loss - loss_val) < 1e-4:
        impatient += 1
        if loss_val < best_loss:
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())
    else:
        best_loss = loss_val
        best_model = deepcopy(model.state_dict())
        impatient = 0

    if impatient >= patience:
        print(f'Breaking due to early stopping at epoch {epoch}')
        break

    epoch_train_loss = epoch_train_loss / training_events

    if epoch % display_step == 0:
        print(f"Epoch {epoch:4d}: Training loss = {epoch_train_loss.item():.4f}, loss_val = {loss_val:.4f}")

# Evaluation
model.load_state_dict(best_model)
model.eval()

# All training & testing sequences stacked into a single batch
with torch.no_grad():
    final_loss_train = aggregate_loss_over_dataloader(dl_train)
    final_loss_val = aggregate_loss_over_dataloader(dl_val)
    final_loss_test = aggregate_loss_over_dataloader(dl_test)

print(f'Negative log-likelihood:\n'
      f' - Train: {final_loss_train:.4f}\n'
      f' - Val:   {final_loss_val:.4f}\n'
      f' - Test:  {final_loss_test:.4f}')

actual_times, predicted_times, actual_marks, predicted_marks = get_prediction_for_all_events(model, dl_test)
all_RMSE = (((predicted_times - actual_times) / actual_times) ** 2).mean().sqrt()
all_f1= f1_score(predicted_marks.detach().numpy(),actual_marks.detach().numpy(),average ='micro')
actual_times, predicted_times, actual_marks, predicted_marks = get_prediction_for_last_events(model, dl_test)
last_RMSE = (((predicted_times - actual_times) / actual_times) ** 2).mean().sqrt()
last_f1= f1_score(predicted_marks.detach().numpy(),actual_marks.detach().numpy(),average ='micro')


print('All event RMSE:{} ,last event RMSE {}'.format(all_RMSE.item(), last_RMSE.item()))
print('All event Accuracy:{} ,last event Accuracy {}'.format(all_f1, last_f1))

# torch.save(model.state_dict(), 'intensity_free_model')
