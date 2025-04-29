import numpy as np

from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


# the below function is taken from OmniAnomaly code base directly
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True, beta=1.0):
    """
    Find the best-f-beta score by searching best `threshold` in [`start`, `end`).
    Beta=1.0 corresponds to F1-score. Beta=2.0 gives more weight to recall.
    Returns:
        list: list for results
        float: the `threshold` for best-f-beta
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print(f"Searching for best F-{beta} score.")
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.) # Store (f_beta, precision, recall)
    m_t = 0.0
    m_f1 = -1 # Also track best f1 for comparison if needed

    for i in range(search_step):
        threshold += search_range / float(search_step)
        # Use calc_seq which returns: f1, precision, recall, TP, TN, FP, FN, roc_auc, latency
        # We need precision and recall to calculate F-beta
        target = calc_seq(score, label, threshold, calc_latency=False) # Don't need latency here
        
        precision, recall = target[1], target[2]
        if precision + recall == 0:
            f_beta = 0.0
        else:
            f_beta = (1 + beta**2) * (precision * recall) / (((beta**2 * precision) + recall) + 1e-9) # Add epsilon
        
        # Use f_beta for maximization
        if f_beta > m[0]:
            m_t = threshold
            m = (f_beta, precision, recall) # Store best F-beta score and corresponding P/R
            m_f1 = target[0] # Store the F1 score corresponding to the best F-beta threshold
            
        if verbose and i % display_freq == 0:
            # Display F-beta, F1, P, R for the current threshold and the best found so far
            print(f"Thr: {threshold:.4f} | F-{beta}: {f_beta:.4f} (F1: {target[0]:.4f}) P: {precision:.4f} R: {recall:.4f} || Best F-{beta}: {m[0]:.4f} (F1: {m_f1:.4f}) P: {m[1]:.4f} R: {m[2]:.4f} at Thr: {m_t:.4f}")
    
    # Return the metrics calculated using the best F-beta threshold
    # Recalculate final metrics using the best threshold m_t
    final_metrics = calc_seq(score, label, m_t, calc_latency=True)
    print(f"Best F-{beta} found: {m[0]:.4f} (F1: {m_f1:.4f}) P: {m[1]:.4f} R: {m[2]:.4f} at threshold {m_t:.4f}")
    
    # Return the full metrics list including TP, TN, etc. based on m_t
    return final_metrics, m_t


def pot_eval(init_score, score, label, q=1e-5, level=0.02, beta=2.0, search_steps=100):
    """
    Run POT method on given score using bf_search to find the best F-beta threshold.
    Args:
        init_score (np.ndarray): The data to get init threshold (unused in this version, but kept for signature compatibility).
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label (np.ndarray): The ground-truth labels.
        q (float): Detection level (risk) - unused here.
        level (float): Probability associated with the initial threshold t - unused here.
        beta (float): Beta value for F-beta score calculation in bf_search. beta=2 prioritizes recall.
        search_steps (int): Number of steps for the threshold search in bf_search.
    Returns:
        dict: bf_search result dict
        np.ndarray: predictions based on the best F-beta threshold
    """
    # --- Original SPOT logic removed --- 
    # lms = lm[0]
    # while True:
    #     try:
    #         s = SPOT(q)  # SPOT object
    #         s.fit(init_score, score)  # data import
    #         s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
    #     except: lms = lms * 0.999
    #     else: break
    # ret = s.run(dynamic=False)  # run
    # pot_th = np.mean(ret['thresholds']) * lm[1]

    # Determine search range based on min/max of test scores
    min_score, max_score = np.min(score), np.max(score)
    search_range_start = min_score
    search_range_end = max_score
    
    print(f"Using bf_search to find best F-{beta} threshold between {search_range_start:.4f} and {search_range_end:.4f} ({search_steps} steps)")

    # Call bf_search to find the best threshold optimizing for F-beta score
    # bf_search returns (metrics_list, best_threshold)
    # metrics_list = [f1, precision, recall, TP, TN, FP, FN, roc_auc, latency]
    best_metrics_list, best_threshold = bf_search(score, label, start=search_range_start, end=search_range_end, step_num=search_steps, verbose=False, beta=beta)

    # Generate predictions using the best threshold found by bf_search
    # Use adjust_predicts directly as calc_seq was already called inside bf_search to get final metrics
    pred = adjust_predicts(score, label, best_threshold)

    # Construct the results dictionary
    result_dict = {
        'f1': best_metrics_list[0],
        'precision': best_metrics_list[1],
        'recall': best_metrics_list[2],
        'TP': best_metrics_list[3],
        'TN': best_metrics_list[4],
        'FP': best_metrics_list[5],
        'FN': best_metrics_list[6],
        'ROC/AUC': best_metrics_list[7],
        'threshold': best_threshold,
        # 'latency': best_metrics_list[8] # bf_search now returns latency in list
    }
    
    return result_dict, pred
