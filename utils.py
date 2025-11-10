import torch
from safetensors import safe_open
import numpy as np
import json
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import yaml


def parse_task(dataset_name):
    if 'opt' in dataset_name:
        return 'opt'
    elif 'stability' in dataset_name:
        return 'stability'
    elif 'range' in dataset_name:
        return 'range'
    else:
        raise ValueError('Invalid dataset name')

def load_config(config_path):
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

def save_config(configs, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(configs, file, default_flow_style=False, sort_keys=False)



def read_fasta(fasta, return_as_dict=False):
    headers, sequences = [], []
    with open(fasta, 'r') as fast:

        for line in fast:
            if line.startswith('>'):
                head = line.replace('>', '').strip()
                headers.append(head)
                sequences.append('')
            else:
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq

    if return_as_dict:
        return dict(zip(headers, sequences))
    else:
        return (headers, sequences)


def write_fasta(headers, seqdata, path):
    with open(path, 'w') as pp:
        for i in range(len(headers)):
            pp.write('>' + headers[i] + '\n' + seqdata[i] + '\n')

    return


def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def write_json(writedict, path, indent=4, sort_keys=False):
    f = open(path, 'w')
    _ = f.write(json.dumps(writedict, indent=indent, sort_keys=sort_keys, default=convert_to_serializable))
    f.close()


def read_json(path):
    f = open(path, 'r')
    readdict = json.load(f)
    f.close()

    return readdict


def replace_noncanonical(seq, replace_char='X'):
    '''Replace all non-canonical amino acids with a specific character'''

    for char in ['B', 'J', 'O', 'U', 'Z']:
        seq = seq.replace(char, replace_char)
    return seq


def get_amino_composition(seq, normalize=True):
    '''Return the amino acid composition for a protein sequence'''

    aac = np.array([seq.count(amino) for amino in list('ACDEFGHIKLMNPQRSTVWY')])
    if normalize:
        aac = aac / len(seq)

    return aac


# freeze the model parameters
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


#count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_weight(model, checkpoint_path, strict=False):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict, strict=strict)


def load_weight_part(model, checkpoint_path, part):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            if part == key.split('.')[0]:
                state_dict[key] = f.get_tensor(key)
    print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)


# init model weight xavier
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.size()) > 1:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            torch.nn.init.zeros_(param.data)


def metrics_cls(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def metrics_reg(eval_pred):
    predictions, targets = eval_pred
    # R? (Coefficient of Determination)
    r2 = r2_score(targets, predictions)
    # Pearson Correlation Coefficient
    pearson_corr = np.corrcoef(predictions, targets)[0, 1]
    # Spearman Correlation Coefficient
    spearman_corr = np.corrcoef(rankdata(predictions), rankdata(targets))[0, 1]
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    #RMSE
    rmse = np.sqrt(mean_squared_error(targets, predictions))

    return {
        'R2': float(r2),
        'Pearson Correlation': float(pearson_corr),
        'Spearman Correlation': float(spearman_corr),
        'MAE': float(mae),
        'RMSE': float(rmse)
    }


def plot_roc(models_pred, y_true, n_classes, save_dir):
    plt.figure()
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(models_pred))]
    colors = cycle(colors)
    for model_name, y_pred in models_pred.items():
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[*range(n_classes)])
        y_pred_bin = label_binarize(y_pred, classes=[*range(n_classes)])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.plot(fpr["macro"], tpr["macro"], color=colors.__next__(),
                 label=model_name + ' macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('macro-average ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/ROC_Curve.png')
    plt.show()


def plot_pr(models_pred, y_true, n_classes, save_dir):
    plt.figure()
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'yellow', 'purple', 'pink'])
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(models_pred))]
    colors = cycle(colors)
    for model_name, y_pred in models_pred.items():
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=[*range(n_classes)])
        y_pred_bin = label_binarize(y_pred, classes=[*range(n_classes)])

        # Compute PR curve and PR area for each class
        precision = dict()
        recall = dict()
        pr_auc = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

        # Compute macro-average PR curve and PR area
        all_precision = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))
        mean_recall = np.zeros_like(all_precision)
        for i in range(n_classes):
            mean_recall += np.interp(all_precision, precision[i], recall[i])
        mean_recall /= n_classes

        precision["macro"] = all_precision
        recall["macro"] = mean_recall
        pr_auc["macro"] = auc(recall["macro"], precision["macro"])

        # Plot all PR curves
        plt.plot(recall["macro"], precision["macro"], color=colors.__next__(),
                 label=model_name + ' macro-average PR curve (area = {0:0.2f})'
                       ''.format(pr_auc["macro"]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('macro-average PR curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/PR_curve.png')
    plt.show()


def plot_barh(models_metrics, save_dir=None):
    # models_metrics: dict {'model_name1': [acc, precision, recall, f1], 'model_name2': [acc, precision, recall, f1], ...}

    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    for idx, metric in enumerate(metrics):
        plt.figure(figsize=(8, 5))
        x = np.array([_ for _ in models_metrics.keys()])
        y = np.array([model_metrics[idx] for model_metrics in models_metrics.values()])
        # set the color of the bar
        colors = plt.cm.plasma(np.linspace(0, 1, len(x)))
        plt.barh(x, y, color=colors)
        plt.title(metric)
        plt.xlim([0.0, 1.05])
        # display the value of the bar
        for i, v in enumerate(y):
            plt.text(v, i, f'{v:.4f}', color='black', va='center')

        if save_dir is not None:
            plt.savefig(f'{save_dir}/{metric}.png', bbox_inches='tight')
        plt.show(bbox_inches='tight')


def scatter_plot_with_density(x, y, xlabel, ylabel, save_dir, title='Scatter_Plot'):
    # Calculate point density
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # Sort the points by density, so higher density points are plotted on top
    idx = density.argsort()
    x, y, density = x[idx], y[idx], density[idx]

    # Plot the scatter plot with density colormap
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=density, s=20, cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([np.min(x), np.max(x)])
    plt.ylim([np.min(x), np.max(x)])
    plt.title(title)
    if 'PatchET' in title:
        title = 'MAE_low' if 'low' in title else 'MAE_high'
    plt.savefig(f'{save_dir}/{title}.pdf', bbox_inches='tight')


def interval_evaluation_opt(labels, predictions):
    index_25 = np.where(labels < 25)[0]
    index25_50 = np.where((labels >= 25) & (labels < 50))[0]
    index50_80 = np.where((labels >= 50) & (labels < 80))[0]
    index80_ = np.where(labels>=80)[0]
    index_dict = {'<25': index_25, '25-50': index25_50, '50-80': index50_80, '>80': index80_}
    metrics_interval = {}
    for k, v in index_dict.items():
        predictions_interval = predictions[v]
        labels_interval = labels[v]
        metrics = metrics_reg((predictions_interval, labels_interval))
        metrics_interval[k] = metrics
    return metrics_interval


def interval_evaluation_stability(labels, predictions):
    index_45 = np.where(labels < 45)[0]
    index45_70 = np.where((labels >= 45) & (labels < 70))[0]
    index70_ = np.where(labels >= 70)[0]

    index_dict = {'<45': index_45, '45-70': index45_70, '>70': index70_}
    metrics_interval = {}
    for k, v in index_dict.items():
        predictions_interval = predictions[v]
        labels_interval = labels[v]
        metrics = metrics_reg((predictions_interval, labels_interval))
        metrics_interval[k] = metrics
    return metrics_interval


def plot_interval_evaluation(metrics_interval, save_dir):
    x = metrics_interval.keys()
    y = [metrics_interval[k]['MAE'] for k in x]
    plt.figure()
    plt.bar(x, y)

    # Display the value of the bar
    for i, v in enumerate(y):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

    plt.xlabel('Temperature Interval')
    plt.ylabel('MAE')
    plt.title('MAE for different temperature intervals')
    plt.savefig(f'{save_dir}/MAE_for_different_temperature_intervals.png')


def metrics_range(eval_pred):
    """
    Compute the mean overlap ratio between the predicted and true temperature ranges using NumPy.

    Args:
        eval_pred: Tuple (pred, label)
            - pred: ndarray of shape (batch_size, 2), where pred[:, 0] is T_low and pred[:, 1] is T_high
            - label: ndarray of shape (batch_size, 2), where label[:, 0] is T_low and label[:, 1] is T_high

    Returns:
        dict: {'mean_overlap_ratio': mean_overlap_ratio}
    """
    pred, label = eval_pred
    pred_low, pred_high = pred[:, 0], pred[:, 1]
    label_low, label_high = label[:, 0], label[:, 1]

    mae_low = np.mean(np.abs(pred_low - label_low))
    mae_high = np.mean(np.abs(pred_high - label_high))

    # Calculate intersection range
    inter_low = np.maximum(pred_low, label_low)
    inter_high = np.minimum(pred_high, label_high)

    # Calculate overlap length (if inter_high > inter_low, otherwise zero)
    intersection = np.maximum(inter_high - inter_low, 0)

    # Calculate ground truth range length
    label_range = label_high - label_low

    # Avoid division by zero (if label_range is zero, set overlap to zero)
    overlap_ratio = np.where(label_range > 0, intersection / label_range, 0)

    # Compute mean overlap ratio
    mean_overlap_ratio = np.mean(overlap_ratio)
    return {'mean_overlap_ratio': mean_overlap_ratio, 'mae_low': mae_low, 'mae_high': mae_high}


def overlap_ratio(pred, label, save_dir):
    pred_low, pred_high = pred[:, 0], pred[:, 1]
    label_low, label_high = label[:, 0], label[:, 1]
    # Calculate intersection range
    inter_low = np.maximum(pred_low, label_low)
    inter_high = np.minimum(pred_high, label_high)

    # Calculate overlap length (if inter_high > inter_low, otherwise zero)
    intersection = np.maximum(inter_high - inter_low, 0)

    # Calculate ground truth range length
    label_range = label_high - label_low

    # Avoid division by zero (if label_range is zero, set overlap to zero)
    overlap_ratio = np.where(label_range > 0, intersection / label_range, 0)

    # Save the overlap ratio to a file
    np.save(f'{save_dir}/overlap_ratio.npy', overlap_ratio)
    return overlap_ratio


def load_metrics(name):
    if name == 'opt':
        return metrics_reg
    elif name == 'stability':
        return metrics_reg
    elif name == 'range':
        return metrics_range
    else:
        raise ValueError('Invalid dataset name')
