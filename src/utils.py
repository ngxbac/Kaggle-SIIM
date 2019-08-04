import numpy as np
import cv2


def post_process(probability, threshold, min_size=3500):

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predict = np.zeros((1024,1024), np.float32)
    num = 0
    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = 1
            num += 1
    return predict, num


def run_length_encode(component):
    component = component.T.flatten()
    start  = np.where(component[1: ] > component[:-1])[0]+1
    end    = np.where(component[:-1] > component[1: ])[0]+1
    length = end-start

    rle = []
    for i in range(len(length)):
        if i==0:
            rle.extend([start[0],length[0]])
        else:
            rle.extend([start[i]-end[i-1],length[i]])

    rle = ' '.join([str(r) for r in rle])
    return rle


def compute_metric(test_truth, test_probability):

    test_num    = len(test_truth)
    truth       = test_truth.reshape(test_num,-1)
    probability = test_probability.reshape(test_num,-1)

    loss = - truth*np.log(probability) - (1-truth)*np.log(1-probability)
    loss = loss.mean()

    t = (truth>0.5).astype(np.float32)
    p = (probability>0.5).astype(np.float32)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    neg_index = np.where(t_sum==0)[0]
    pos_index = np.where(t_sum>=1)[0]

    dice_neg = (p_sum == 0).astype(np.float32)
    dice_pos = 2* (p*t).sum(-1)/((p+t).sum(-1)+1e-12)
    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]
    dice     = np.concatenate([dice_pos,dice_neg])

    dice_neg = np.nan_to_num(dice_neg.mean().item(),0)
    dice_pos = np.nan_to_num(dice_pos.mean().item(),0)
    dice = dice.mean()

    return loss, dice, dice_neg, dice_pos


def kaggle_metric_one(predict, truth):

    if truth.sum() ==0:
        if predict.sum() ==0: return 1
        else:                 return 0

    #----
    predict = predict.reshape(-1)
    truth   = truth.reshape(-1)

    intersect = predict*truth
    union     = predict+truth
    dice      = 2.0*intersect.sum()/union.sum()
    return dice


def compute_kaggle_lb(test_truth, test_probability, threshold, min_size):

    test_num    = len(test_truth)

    kaggle_pos = []
    kaggle_neg = []
    for b in range(test_num):
        truth       = test_truth[b,0]
        probability = test_probability[b,0]

        if truth.shape!=(1024,1024):
            truth = cv2.resize(truth, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
            truth = (truth>0.5).astype(np.float32)

        if probability.shape!=(1024,1024):
            probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)

        #-----
        predict, num_component = post_process(probability, threshold, min_size)

        score = kaggle_metric_one(predict, truth)
        print('\r%3d  %s   %0.5f  %0.5f'% (b, predict.shape, probability.mean(), probability.max()), end='', flush=True)

        if truth.sum()==0:
            kaggle_neg.append(score)
        else:
            kaggle_pos.append(score)

    print('')
    kaggle_neg = np.array(kaggle_neg)
    kaggle_pos = np.array(kaggle_pos)
    kaggle_neg_score = kaggle_neg.mean()
    kaggle_pos_score = kaggle_pos.mean()
    kaggle_score = 0.7886*kaggle_neg_score + (1-0.7886)*kaggle_pos_score

    return kaggle_score,kaggle_neg_score,kaggle_pos_score