from sklearn.metrics import roc_auc_score

def compute_auroc(data_gt, data_pred, class_count):
    """ Computes the area under ROC Curve
    data_gt: ground truth data
    data_pred: predicted data
    class_count: Number of classes
    """

    out_auroc_list = []
    data_np_gt = data_gt.cpu().numpy()
    data_np_pred = data_pred.cpu().numpy()
    for i in range(class_count):
        try:
            out_auroc_list.append(roc_auc_score(data_np_gt[:, i], data_np_pred[:, i]))
        except ValueError:
            out_auroc_list.append(0)
    return out_auroc_list
