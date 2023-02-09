from sklearn.metrics import roc_auc_score
import numpy as np
def compute_performance(pred, gt):
    #This function computes the performance metrics : Accuracy, AUC, Sensitivity and Specificity 
    acc_lst=[]
    auc_lst=[]
    sens_lst=[]
    spec_lst=[]
    pred_scr=pred.copy()
    pred_cls=pred.copy()
    idx0=np.where(pred_cls<0.5)
    idx1=np.where(pred_cls>=0.5)
    pred_cls[idx0]=0
    pred_cls[idx1]=1
    
    for cls in range(0, pred_scr.shape[1]):    
        tmp_prd_scr=pred_scr[:,cls]
        tmp_prd_cls=pred_cls[:, cls]
        tmp_gt=gt[:, cls]
        
        TP=np.where((tmp_gt==1) & (tmp_prd_cls==1))[0].shape[0]
        TN=np.where((tmp_gt==0) & (tmp_prd_cls==0))[0].shape[0]
        FP=np.where((tmp_gt==0) & (tmp_prd_cls==1))[0].shape[0]
        FN=np.where((tmp_gt==1) & (tmp_prd_cls==0))[0].shape[0]
        
        acc=(TP+TN)/(TP+TN+FP+FN)
        sens=TP/(TP+FN)
        spec=TN/(TN+FP)
        auc=roc_auc_score(tmp_gt, tmp_prd_scr)
        
        sens_lst.append(sens)
        spec_lst.append(spec)
        acc_lst.append(acc)
        auc_lst.append(auc)
    
    sens_lst=np.array(sens_lst)
    spec_lst=np.array(spec_lst)
    acc_lst=np.array(acc_lst)
    auc_lst=np.array(auc_lst)
    return sens_lst, spec_lst, acc_lst, auc_lst
    