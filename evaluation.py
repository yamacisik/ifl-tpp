import torch
import torch.nn.functional as F

from typing import Any, List, Optional


def get_prediction_for_all_events(model, dl):
    """
    Function for calculating the predictions for every arrival times and marks for each event in each sequence.
    Returns a tuple of actual event times, predicted event times, actual marks and predicted event types.

    Parameters
    ----------
    model
    dl

    Returns
    -------

    """

    total_num_events = dl.dataset.total_num_events
    all_event_time_predictions = []
    all_event_time_values = []
    all_actual_marks = []
    all_predicted_marks = []

    for batch in dl:
        lengths = batch.mask.sum(-1) - 1  ## Minus 1 because they also calculate the survival for the end of time.

        features = model.get_features(batch)
        context = model.get_context(features)
        inter_time_dist = model.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)

        ## Arrival Time Prediction
        predicted_times = inter_time_dist.mean
        all_predicted_times = \
            torch.nn.utils.rnn.pack_padded_sequence(predicted_times.T, lengths.cpu(), batch_first=False,
                                                    enforce_sorted=False)[0]
        all_actual_times = \
            torch.nn.utils.rnn.pack_padded_sequence(inter_times.T, lengths.cpu(), batch_first=False, enforce_sorted=False)[0]
        all_event_time_values.append(all_actual_times[:-1])
        all_event_time_predictions.append(all_predicted_times[:-1])

        #         ## Mark Prediction
        predicted_marks = torch.log_softmax(model.mark_linear(context), dim=-1).argmax(-1)
        predicted_marks = \
            torch.nn.utils.rnn.pack_padded_sequence(predicted_marks.T, lengths.cpu(), batch_first=False,
                                                    enforce_sorted=False)[0]
        actual_marks = \
            torch.nn.utils.rnn.pack_padded_sequence(batch.marks.T, lengths.cpu(), batch_first=False, enforce_sorted=False)[0]
        all_actual_marks.append(actual_marks)
        all_predicted_marks.append(predicted_marks)

    #         all_event_accuracy = (predicted_marks ==batch.marks)*batch.mask
    #         all_event_accuracies.append(all_event_accuracy)
    #         last_event_accuracy = all_event_accuracy[x_index,y_index]
    #         last_event_accuracies.append(last_event_accuracy)

    #     last_event_rmse = (torch.cat(last_event_errors,axis = 0)**2).mean().sqrt()
    #     all_event_rmse = ((torch.cat(total_errors,-1)**2).sum()/total_num_events).sqrt()
    #     all_event_accuracy = torch.cat(all_event_accuracies,-1).sum()/total_num_events
    #     last_event_accuracy = torch.cat(last_event_accuracies,0)
    # #     last_event_accuracy = last_event_accuracy.sum()/len(last_event_accuracy)

    #     return last_event_rmse,all_event_rmse,all_event_accuracy,last_event_accuracy

    all_event_time_values = torch.cat(all_event_time_values)
    all_event_time_predictions = torch.cat(all_event_time_predictions)
    all_actual_marks = torch.cat(all_actual_marks)
    all_predicted_marks = torch.cat(all_predicted_marks)

    return all_event_time_values, all_event_time_predictions, all_actual_marks, all_predicted_marks


def get_prediction_for_last_events(model, dl):
    """
    Function for calculating the predictions for last event arrival times and marks for each sequence.
    Returns a tuple of actual event times, predicted event times, actual marks and predicted event types.

    Parameters
    ----------
    model
    dl

    Returns
    -------

    """
    event_time_predictions = []
    event_time_values = []
    actual_marks = []
    predicted_marks = []

    for batch in dl:
        y_index = batch.mask.sum(
            -1).long() - 1  ## Minus 1 because they also calculate the survival for the end of time.
        features = model.get_features(batch)
        context = model.get_context(features)
        inter_time_dist = model.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)
        x_index = torch.arange(0, len(inter_times))

        ## Arrival Time Prediction
        actual_time = inter_times[x_index, y_index]
        predicted_time = inter_time_dist.mean[x_index, y_index]

        ## Mark Prediction
        actual_mark = batch.marks[x_index, y_index]
        predicted_mark = torch.log_softmax(model.mark_linear(context), dim=-1).argmax(-1)[x_index, y_index]

        event_time_predictions.append(predicted_time)
        event_time_values.append(actual_time)
        actual_marks.append(actual_mark)
        predicted_marks.append(predicted_mark)

    event_time_values = torch.cat(event_time_values)
    event_time_predictions = torch.cat(event_time_predictions)
    actual_marks = torch.cat(actual_marks)
    predicted_marks = torch.cat(predicted_marks)

    return event_time_values, event_time_predictions, actual_marks, predicted_marks

