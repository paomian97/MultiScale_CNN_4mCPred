import numpy as np

# Read positive sample data from the training set
def Load_train_positive():
    with open(file=Path_train_positive, mode='r') as file:
        file = file.read().strip().split('\n')
        file = file[1:]
        seqs = []
        labels = []
        for line in file:
            seq, label = line.split(',')
            seqs.append(seq)
            labels.append(np.int32(label))
        np.save(r'./data_processed/train_seq_positive', seqs)
        np.save(r'./data_processed/train_label_positive', labels)

    return seqs, labels

# Read negative sample data from the training set
def Load_train_negative():
    with open(file=Path_train_negative, mode='r') as file:
        file = file.read().strip().split('\n')
        file = file[1:]
        seqs = []
        labels = []
        for line in file:
            seq, label = line.split(',')
            seqs.append(seq)
            labels.append(np.int32(label))
        np.save(r'./data_processed/train_seq_negative', seqs)
        np.save(r'./data_processed/train_label_negative', labels)

    return seqs, labels

# Read positive sample data from the testing set
def Load_test_positive():
    with open(file=Path_test_positive, mode='r') as file:
        file = file.read().strip().split('\n')
        file = file[1:]
        seqs = []
        labels = []
        for line in file:
            seq, label = line.split(',')
            seqs.append(seq)
            labels.append(np.int32(label))
        np.save(r'./data_processed/test_seq_positive', seqs)
        np.save(r'./data_processed/test_label_positive', labels)

    return seqs, labels


# Read negative sample data from the testing set
def Load_test_negative():
    with open(file=Path_test_negative, mode='r') as file:
        file = file.read().strip().split('\n')
        file = file[1:]
        seqs = []
        labels = []
        for line in file:
            seq, label = line.split(',')
            seqs.append(seq)
            labels.append(label)
        np.save(r'./data_processed/test_seq_negative', seqs)
        np.save(r'./data_processed/test_label_negative', labels)

    return seqs, labels

if __name__ == "__main__":
    Path_train_positive = r'./dataset/Positive_train.txt'
    Path_train_negative = r'./dataset/Negative_train.txt'
    Path_test_positive = r'./dataset/Positive_Independent.txt'
    Path_test_negative = r'./dataset/Negative_Independent.txt'

    train_seq_positive, train_label_positive = Load_train_positive()
    train_seq_negative, train_label_negative = Load_train_negative()
    test_seq_positive, test_label_positive = Load_test_positive()
    test_seq_negative, test_label_negative = Load_test_negative()

    print('The processed data has been stored in the data_processed folder')
    print('Output the relevant data size:')
    print('Train_sequences_positive and Train_labels_positive size:', len(train_seq_positive), len(train_label_positive))
    print('Train_sequences_negative and Train_labels_negative size:', len(train_seq_negative), len(train_label_negative))
    print('Test_sequences_positive and Test_labels_positive size:', len(test_seq_positive), len(test_label_positive))
    print('Test_sequences_negative and Test_labels_negative size:', len(test_seq_negative), len(test_label_negative))





