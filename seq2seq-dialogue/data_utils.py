# -*- coding: UTF-8 -*-

"""Utilities for downloading data, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import jieba

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize(cn).
_SEN_SPLIT = re.compile(b"([。，！？、\"'：；）（])")
_DIGIT_RE = re.compile(br"\d")

# Regular expressions used to tokenize(en).
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")

# data files
infile = '/movie_in.txt'
outfile = '/movie_out.txt'


def jieba_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        for simple_sentence in _SEN_SPLIT.split(space_separated_fragment):
            words.extend(jieba.cut(simple_sentence))
    return [w for w in words if w]


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 50000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def append_vocabulary(vocabulary_path, append_data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Extend vocabulary file (if it already exists) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      append_data_path: data file that will be used to extend vocabulary.
      max_vocabulary_size: limit on the size of the final vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if gfile.Exists(vocabulary_path):
        vocab, rev_vocab = initialize_vocabulary(vocabulary_path)
        print("Extending vocabulary %s from data %s" % (vocabulary_path, append_data_path))
        with gfile.GFile(append_data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 50000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 50000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                      tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_dlg_data(data_dir, vocabulary_size, tokenizer=None):
    """Get dialogue data into data_dir, create vocabularies and tokenize data.

    Args:
      data_dir: directory in which the data sets will be stored.
      vocabulary_size: size of the Chinese dialogue vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for question training data-set,
        (2) path to the token-ids for answer training data-set,
        (3) path to the token-ids for question development data-set,
        (4) path to the token-ids for answer development data-set,
        (5) path to the vocabulary file,
    """
    from_train_path = data_dir + infile
    to_train_path = data_dir + outfile
    # from_dev_path = dev_path + ".ques"
    # to_dev_path = dev_path + ".ans"
    return prepare_data(data_dir, from_train_path, to_train_path,
                        # from_dev_path, to_dev_path,
                        vocabulary_size,tokenizer)


def prepare_data(data_dir, from_train_path, to_train_path,
                 # from_dev_path, to_dev_path,
                 vocabulary_size, tokenizer=None):
    """Preapre all necessary files that are required for the training.

      Args:
        data_dir: directory in which the data sets will be stored.
        from_train_path: path to the file that includes "from" training samples.
        to_train_path: path to the file that includes "to" training samples.
        from_dev_path: path to the file that includes "from" dev samples.
        to_dev_path: path to the file that includes "to" dev samples.
        vocabulary_size: size of the dialogue vocabulary to create and use.
        tokenizer: a function to use to tokenize each data sentence;
          if None, basic_tokenizer will be used.

      Returns:
        A tuple of 6 elements:
          (1) path to the token-ids for "from language" training data-set,
          (2) path to the token-ids for "to language" training data-set,
          (3) path to the token-ids for "from language" development data-set,
          (4) path to the token-ids for "to language" development data-set,
          (5) path to the "from language" vocabulary file,
          (6) path to the "to language" vocabulary file.
      """
    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab%d.qa" % vocabulary_size)
    create_vocabulary(vocab_path, from_train_path, vocabulary_size, tokenizer)
    append_vocabulary(vocab_path, to_train_path, vocabulary_size, tokenizer)

    # Create token ids for the training data.
    to_train_ids_path = to_train_path + (".ids%d" % vocabulary_size)
    from_train_ids_path = from_train_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(to_train_path, to_train_ids_path, vocab_path, tokenizer)
    data_to_token_ids(from_train_path, from_train_ids_path, vocab_path, tokenizer)

    # Create token ids for the development data.
    # to_dev_ids_path = to_dev_path + (".ids%d" % vocabulary_size)
    # from_dev_ids_path = from_dev_path + (".ids%d" % vocabulary_size)
    # data_to_token_ids(to_dev_path, to_dev_ids_path, vocab_path, tokenizer)
    # data_to_token_ids(from_dev_path, from_dev_ids_path, vocab_path, tokenizer)

    return (from_train_ids_path, to_train_ids_path,
            # from_dev_ids_path, to_dev_ids_path,
            vocab_path)

    # 基本步骤
    # Special vocabulary symbols - we always put them at the start.（特殊标记，比如eos（end of string），unk（unkown）等）
    # Regular expressions used to tokenize.(包括标点，数字等）
    # Download data
    # tokenize分词（jieba）
    # 词典的创建与初始化（注意词典的结构，不仅有词，还应该有一一映射的id，词典中的词应该经过筛选，如根据词频等，去掉不常见或无意义的词一类的，nltk）
    # 利用词典将输入数据转化为token-id数据
    # 上面后面的函数主要针对特定数据及翻译任务，对于对话元数据的处理而言，建立token-id数据就基本可以了，剩余步骤参考word2vec_basic.py
