import tensorflow as tf
import numpy as np

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
    print('test')
