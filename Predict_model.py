import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

def sn_sp_acc_mcc(true_label, predict_label, pos_label=1):
    import math
    pos_num = np.sum(true_label == pos_label)
    print('pos_num=', pos_num)
    neg_num = true_label.shape[0] - pos_num
    print('neg_num=', neg_num)
    tp = np.sum((true_label == pos_label) & (predict_label == pos_label))
    print('tp=', tp)
    tn = np.sum(true_label == predict_label) - tp
    print('tn=', tn)
    sn = tp / pos_num
    sp = tn / neg_num
    acc = (tp + tn) / (pos_num + neg_num)
    fn = pos_num - tp
    fp = neg_num - tn
    print('fn=', fn)
    print('fp=', fp)

    tp = np.array(tp, dtype=np.float64)
    tn = np.array(tn, dtype=np.float64)
    fp = np.array(fp, dtype=np.float64)
    fn = np.array(fn, dtype=np.float64)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)))
    return sn, sp, acc, mcc

def numerical_transform(sequences):
    unit = [x for x in 'AGCT']
    Dict = {Key:value for value, Key in enumerate(unit)}
    outcome = []
    for seq in sequences:
        outcome.append([Dict[a] for a in seq])
    return np.array(outcome)

def model_predict(sequences):
    feature = numerical_transform(sequences)

    model = tf.keras.models.load_model(r'./model_weights')

    predict = model.predict(feature)
    predict = np.squeeze(predict, axis=-1)
    predict = np.int32(predict > 0.5)
    return predict

if __name__ == '__main__':
    print('Load test set data and make predictions:')

    test_positive_seq = np.load(r'./data_processed/test_seq_positive.npy')
    test_negative_seq = np.load(r'./data_processed/test_seq_negative.npy')
    test_label_positive = np.load(r'./data_processed/test_label_positive.npy')
    test_label_negative = np.load(r'./data_processed/test_label_negative.npy')

    test_data = np.concatenate([test_positive_seq, test_negative_seq], axis=0)
    test_label = np.concatenate([test_label_positive, test_label_negative], axis=0)

    predict_outcome = model_predict(test_data)

    sn, sp, acc, mcc = sn_sp_acc_mcc(test_label, predict_outcome)
    print('Accuracy = ', acc)
    print('Sensitivity = ', sn)
    print('Specificity = ', sp)
    print('Matthews correlation coefficient = ', mcc)
