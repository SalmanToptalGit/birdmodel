import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import pandas.api.types
from typing import Union

from sklearn.metrics import f1_score, precision_score, recall_score

class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def treat_as_participant_error(error_message: str, solution: Union[pd.DataFrame, np.ndarray]) -> bool:
    ''' Many metrics can raise more errors than can be handled manually. This function attempts
    to identify errors that can be treated as ParticipantVisibleError without leaking any competition data.

    If the solution is purely numeric, and there are no numbers in the error message,
    then the error message is sufficiently unlikely to leak usable data and can be shown to participants.

    We expect this filter to reject many safe messages. It's intended only to reduce the number of errors we need to manage manually.
    '''
    # This check treats bools as numeric
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all([pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
        solution_has_bools = any([pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values])
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if 'true' in error_message.lower() or 'false' in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    '''
    Call score. If that raises an error and that already been specifically handled, just raise it.
    Otherwise make a conservative attempt to identify potential participant visible errors.
    '''
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == 'ParticipantVisibleError':
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == 'HostVisibleError':
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err
    return score_result


def verify_valid_probabilities(df: pd.DataFrame, df_name: str):
    """ Verify that the dataframe contains valid probabilities.

    The dataframe must be limited to the target columns; do not pass in any ID columns.
    """
    if not pandas.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(f'All target values in {df_name} must be numeric')

    if df.min().min() < 0:
        raise ParticipantVisibleError(f'All target values in {df_name} must be at least zero')

    if df.max().max() > 1:
        raise ParticipantVisibleError(f'All target values in {df_name} must be no greater than one')

    if not np.allclose(df.sum(axis=1), 1):
        raise ParticipantVisibleError(f'Target values in {df_name} do not add to one within all rows')
#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
import pandas.api.types

import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype  for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0


    return safe_call_score(sklearn.metrics.roc_auc_score, solution[scored_columns].values, submission[scored_columns].values, average='macro')


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution.drop(["row_id"], axis=1, errors="ignore")
    submission = submission.drop(["row_id"], axis=1, errors="ignore")
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for j in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = (
        pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    )
    padded_submission = (
        pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    )
    score = average_precision_score(
        padded_solution.values.astype(int),
        padded_submission.values,
        average="macro",
    )
    return score


def padded_cmap_numpy(y_true, y_pred, padding_factor=5):
    y_true = np.pad(y_true, ((0, padding_factor), (0, 0)), constant_values=1)
    y_pred = np.pad(y_pred, ((0, padding_factor), (0, 0)), constant_values=1)
    return average_precision_score(
        y_true.astype(int),
        y_pred,
        average="macro",
    )

def map_score(solution, submission):
    solution = solution
    submission = submission
    score = average_precision_score(
        solution.values,
        submission.values,
        average='micro',  # 'macro'
    )
    return score

def calculate_metrics(gt, preds):
    f1 = f1_score(y_true=gt, y_pred=np.argmax(preds, axis=1), average='macro')
    precision = precision_score(y_true=gt, y_pred=np.argmax(preds, axis=1), average='macro')
    recall = recall_score(y_true=gt, y_pred=np.argmax(preds, axis=1), average='macro')

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def calculate_competition_metrics(gt, preds, target_columns, one_hot=True):
    if not one_hot:
        ground_truth = np.argmax(gt, axis=1)
        gt = np.zeros((ground_truth.size, len(target_columns)))
        gt[np.arange(ground_truth.size), ground_truth] = 1

    val_df = pd.DataFrame(gt, columns=target_columns)
    pred_df = pd.DataFrame(preds, columns=target_columns)

    cmAP_1 = padded_cmap(val_df, pred_df, padding_factor=1)
    cmAP_5 = padded_cmap(val_df, pred_df, padding_factor=5)
    mAP = map_score(val_df, pred_df)

    val_df['id'] = [f'id_{i}' for i in range(len(val_df))]
    pred_df['id'] = [f'id_{i}' for i in range(len(pred_df))]
    train_score = score(val_df, pred_df, row_id_column_name='id')

    return {
        "cmAP_1": cmAP_1,
        "cmAP_5": cmAP_5,
        "mAP": mAP,
        "ROC": train_score,
    }


def metrics_to_string(scores, key_word):
    log_info = ""
    for key in scores.keys():
        log_info = log_info + f"{key_word} {key} : {scores[key]:.4f}, "
    return log_info