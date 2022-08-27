import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

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

    print('Accuracy=', accuracy_score(test_label, predict_outcome))
