import logging

import nltk as nltk
import numpy as np
import torch
from joblib import Memory
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

CACHE_DIR = 'tmp/cache'
MEMORY = Memory(CACHE_DIR, verbose=2)
RANDOM_STATE = 42  # 42 is chosen as the binary representation of the number has the same amount of 1s as 0s 1s


@MEMORY.cache
def load_pretrained_embeddings(path, unk_token=None):
    """
    Load pretrained embeddings from the given path.

    :param path: the path to the embeddings
    :param unk_token: the token to use for unknown words
    """
    vocab = dict()
    vectors = []
    with open(path, mode='r', encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f)):
            word, *vector_str = line.strip().split(' ')
            if len(vector_str) == 1:
                logging.debug(f"[load_pretrained_embeddings] Ignoring row {i + 1}: {line}")
                continue

            vector = torch.tensor([float(x) for x in vector_str])

            vocab[word] = len(vocab)
            vectors.append(vector)

    if unk_token:
        logging.debug(f"Adding UNK token: '{unk_token}'")
        vocab[unk_token] = len(vocab)
        vectors.append(torch.zeros_like(vectors[0]))

    embedding = torch.stack(vectors)

    return vocab, embedding


def build_tokenizer_for_word_embeddings(vocab, unk_token):
    """
    Build a tokenizer for word-level embeddings.
    This method was adapted from https://github.com/lgalke/text-clf-baselines/blob/main/tokenization.py

    :param vocab: the vocabulary to use
    :param unk_token: the token to use for unknown words
    :return: the tokenizer
    """
    model = WordLevel(vocab, unk_token)
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def prepare_data(dataset, tokenizer, dataset_class, *, shuffle=False, max_length=None):
    """
    Prepare data for training by creating a PyTorch dataset.

    :param dataset: the dataset dictionary
    :param tokenizer: the tokenizer to use
    :param dataset_class: the dataset class to use
    :param shuffle: whether to shuffle the data
    :param max_length: the maximum length of the input sequences
    :return: the test and train dataset, the label dictionary
    """
    train_text, train_labels = dataset['train']

    if shuffle:
        # shuffle train_text and train_labels in the same way
        state = np.random.get_state()
        np.random.shuffle(train_text)
        np.random.set_state(state)
        np.random.shuffle(train_labels)

    # tokenize text
    train_encodings = tokenizer(train_text, truncation=True, padding=True, max_length=max_length)
    test_text, test_labels = dataset['test']
    test_encodings = tokenizer(test_text, truncation=True, padding=True, max_length=max_length)
    # encode labels
    label_dict = dataset['label_dict']
    train_labels_encoded = [label_dict[label] for label in train_labels]
    test_labels_encoded = [label_dict[label] for label in test_labels]
    # create dataset
    train_data = dataset_class(train_encodings, train_labels_encoded)
    test_data = dataset_class(test_encodings, test_labels_encoded)

    return test_data, train_data, dataset['label_dict']


def prepare_data_custom_tokenizer(dataset, tokenizer, dataset_class):
    """
    Prepare data for training with a non-callable tokenizer.
    Used for LSTM and MLP when pretrained embeddings are used.

    :param dataset: the dataset dictionary
    :param tokenizer: the tokenizer to use
    :param dataset_class: the dataset class to use

    :return: the test and train dataset, the label dictionary
    """
    train_text, train_labels = dataset['train']
    # tokenize text
    train_encodings = [tokenizer.encode(text) for text in train_text]
    test_text, test_labels = dataset['test']
    test_encodings = [tokenizer.encode(text) for text in test_text]
    # encode labels
    label_dict = dataset['label_dict']
    train_labels_encoded = [label_dict[label] for label in train_labels]
    test_labels_encoded = [label_dict[label] for label in test_labels]
    # create dataset
    train_data = dataset_class(train_encodings, train_labels_encoded)
    test_data = dataset_class(test_encodings, test_labels_encoded)

    return test_data, train_data, dataset['label_dict']


@MEMORY.cache
def load_data(key):
    """
    Load the data for the given key.
    :param key: The name of the dataset.
    :return: A dictionary with the following keys:
        - name: the name of the dataset
        - train: a tuple of (text, labels)
        - test: a tuple of (text, labels)
        - label_dict: a dictionary mapping labels to integers
    """
    dataset = {"name": key}

    if key == 'MR':
        load_MR(dataset)
    elif key == 'R8':
        load_R8(dataset)
    elif key == 'SearchSnippets':
        load_SearchSnippets(dataset)
    elif key == 'Twitter':
        load_Twitter(dataset)
    elif key == 'TREC':
        load_TREC(dataset)
    elif key == 'SST2':
        load_SST2(dataset)
    elif key == 'NICE':
        load_NICE(dataset)
    elif key == 'NICE2':
        load_NICE2(dataset)
    elif key == 'STOPS':
        load_STOPS(dataset)
    elif key == 'STOPS2':
        load_STOPS2(dataset)
    else:
        raise ValueError(f"Unknown dataset: {key}")

    return dataset


def load_STOPS(dataset):
    """
    Load the STOPS dataset
    :param dataset: the dataset dictionary
    """
    load_tab_spaced_data("data/STOPS/STOPS/STOPS_test.txt", "data/STOPS/STOPS/STOPS_train.txt", dataset)


def load_STOPS2(dataset):
    """
    Load the STOPS2 dataset
    :param dataset: the dataset dictionary
    """
    load_tab_spaced_data("data/STOPS/STOPS-2/STOPS-2_test.txt", "data/STOPS/STOPS-2/STOPS-2_train.txt",
                         dataset)


def load_NICE(dataset):
    """
    Load the NICE dataset
    :param dataset: the dataset dictionary
    """
    load_tab_spaced_data("data/NICE/NICE/NICE_test.txt", "data/NICE/NICE/NICE_train.txt", dataset)


def load_NICE2(dataset):
    """
    Load the binary NICE dataset
    :param dataset: the dataset dictionary
    """
    load_tab_spaced_data("data/NICE/NICE_binary/NICE_test.txt", "data/NICE/NICE_binary/NICE_train.txt", dataset)


def load_tab_spaced_data(test_path, train_path, dataset):
    """
    Load tab-separated data from the given path.
    :param test_path: the path to the test data
    :param train_path: the path to the train data
    :param dataset: the dataset dictionary
    """
    # load training data
    with open(train_path, "r", encoding="utf-8") as f:
        # split at \t
        train_data = [line.split("\t") for line in f.read().splitlines()]
    with open(test_path, "r", encoding="utf-8") as f:
        # split at \t
        test_data = [line.split("\t") for line in f.read().splitlines()]

    # create label dictionary
    label_dict = create_dict(set.union(set([data[0] for data in train_data]), set([data[0] for data in test_data])))

    # add to dataset
    dataset["train"] = ([text for label, text in train_data], [label for label, text in train_data])
    dataset["test"] = ([text for label, text in test_data], [label for label, text in test_data])
    dataset["label_dict"] = label_dict


def load_SST2(dataset):
    """
    Load the SST2 dataset.
    :param dataset: the dataset dictionary
    """
    # load training data
    with open("data/sst2/sst2-train.txt", "r", encoding="utf-8") as f:
        # split at \t
        train_data = [line.split("\t") for line in f.read().splitlines()]
    # append dev to training data
    with open("data/sst2/sst2-dev.txt", "r", encoding="utf-8") as f:
        # split at \t
        train_data += [line.split("\t") for line in f.read().splitlines()]
    with open("data/sst2/sst2-test.txt", "r", encoding="utf-8") as f:
        # split at \t
        test_data = [line.split("\t") for line in f.read().splitlines()]

    # create label dictionary
    label_dict = create_dict(set.union(set([data[0] for data in train_data]), set([data[0] for data in test_data])))

    # add to dataset
    dataset["train"] = ([text for label, text in train_data], [label for label, text in train_data])
    dataset["test"] = ([text for label, text in test_data], [label for label, text in test_data])
    dataset["label_dict"] = label_dict


def load_TREC(dataset):
    """
    Load the TREC dataset.
    :param dataset: the dataset dictionary
    """
    # load training data
    with open("data/corpus/TREC.clean.txt", "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    with open("data/TREC/TREC.txt", "r", encoding="utf-8") as f:
        data_information = [(int(word[0]), word[1], word[2]) for word in
                            [line.split() for line in f.read().splitlines()]]

    # train_data/test_data based on data_information (index, split, label)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for index, split, label in data_information:
        if split == 'train':
            train_data.append(data[index])
            train_labels.append(label)
        else:
            test_data.append(data[index])
            test_labels.append(label)

    # create label dictionary
    label_dict = create_dict(set.union(set(train_labels), set(test_labels)))

    # add to dataset
    dataset["train"] = (train_data, train_labels)
    dataset["test"] = (test_data, test_labels)
    dataset["label_dict"] = label_dict


def load_Twitter(dataset):
    """
    Load the Twitter dataset from nltk.
    :param dataset: the dataset dictionary
    """
    from nltk.corpus import twitter_samples

    nltk.data.path.append("data/nltk_data")

    negative = twitter_samples.strings('negative_tweets.json')
    positive = twitter_samples.strings('positive_tweets.json')

    # shuffle with fixed seed for reproducibility
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(negative)
    np.random.shuffle(positive)

    # split into train and test (70%/30%)
    train_size = int(len(negative) * 0.7)
    test_size = len(negative) - train_size
    train_data = negative[:train_size] + positive[:train_size]
    test_data = negative[train_size:] + positive[train_size:]
    train_labels = ["negative"] * train_size + ["positive"] * train_size
    test_labels = ["negative"] * test_size + ["positive"] * test_size

    # build tuples of (text, label)
    train_data = [(text, label) for text, label in zip(train_data, train_labels)]
    test_data = [(text, label) for text, label in zip(test_data, test_labels)]

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    # create label dictionary
    label_dict = create_dict(set.union(set(train_labels), set(test_labels)))

    # add to dataset
    dataset["train"] = ([text for text, label in train_data], [label for text, label in train_data])
    dataset["test"] = ([text for text, label in test_data], [label for text, label in test_data])
    dataset["label_dict"] = label_dict


def load_SearchSnippets(dataset):
    """
    Load the SearchSnippets dataset.
    :param dataset: the dataset dictionary
    """
    # load training data
    with open("data/data-web-snippets/train.txt", "r", encoding='utf8') as f:
        raw_train = [line.strip() for line in f]
    list_of_words = [line.split() for line in raw_train]
    # last element is the label
    train_data = [" ".join(line[:-1]) for line in list_of_words]
    train_labels = [line[-1] for line in list_of_words]
    # load test data
    with open("data/data-web-snippets/test.txt", "r", encoding='utf8') as f:
        raw_train = [line.strip() for line in f]
    list_of_words = [line.split() for line in raw_train]
    # last element is the label
    test_data = [" ".join(line[:-1]) for line in list_of_words]
    test_labels = [line[-1] for line in list_of_words]
    # create label dictionary
    label_dict = create_dict(set.union(set(train_labels), set(test_labels)))
    # add to dataset
    dataset["train"] = (train_data, train_labels)
    dataset["test"] = (test_data, test_labels)
    dataset["label_dict"] = label_dict


def load_R8(dataset):
    """
    Load the R8 dataset.
    :param dataset: the dataset dictionary
    """
    # load training data
    with open("data/R8/train.txt", "r", encoding="utf-8") as f:
        raw_train = [line.strip() for line in f]
    list_of_words = [line.split() for line in raw_train]
    # first element is the label
    train_text = [" ".join(line[1:]) for line in list_of_words]
    train_labels = [line[0] for line in list_of_words]
    # load test data
    with open("data/R8/test.txt", "r", encoding="utf-8") as f:
        raw_test = [line.strip() for line in f]
    list_of_words = [line.split() for line in raw_test]
    # first element is the label
    test_text = [" ".join(line[1:]) for line in list_of_words]
    test_labels = [line[0] for line in list_of_words]
    # create label dictionary
    label_dict = create_dict(set.union(set(train_labels), set(test_labels)))
    # add to dataset
    dataset["train"] = (train_text, train_labels)
    dataset["test"] = (test_text, test_labels)
    dataset["label_dict"] = label_dict


def load_MR(dataset):
    """
    Load the MR dataset.
    :param dataset: the dataset dictionary
    """
    # load training data
    with open("/data/zkxu/MR/label_train.txt", "r", encoding="latin1") as f:
        train_labels = [line.strip() for line in f]
    with open("/data/zkxu/MR/text_train.txt", "r", encoding="latin1") as f:
        train_text = [line.strip() for line in f]

    # load test data
    with open("/data/zkxu/MR/label_test.txt", "r", encoding="latin1") as f:
        test_labels = [line.strip() for line in f]
    with open("/data/zkxu/MR/text_test.txt", "r", encoding="latin1") as f:
        test_text = [line.strip() for line in f]
    # create label dictionary
    label_dict = create_dict(set.union(set(train_labels), set(test_labels)))
    # add to dataset
    dataset["train"] = (train_text, train_labels)
    dataset["test"] = (test_text, test_labels)
    dataset["label_dict"] = label_dict


def create_dict(labels):
    """
    Create a numerical mapping for the given labels.
    :param labels: the labels
    :return: the dictionary
    """
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    return label_dict


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.text.items()}
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, index):
        text = self.text[index]
        label = self.labels[index]
        return text, label

    def __len__(self):
        return len(self.labels)
