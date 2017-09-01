#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from configurable import Configurable
class Example(Configurable):
  """

  """
  def __init__(self, sent, head_words, head_tags, word_tags, *args, **kwargs):
    super(Example, self).__init__(*args, **kwargs)
    self.length = len(sent)
    self.sent = None
    self.data = None
    self.head = None
    self.head_channel = None
    # original word in this setting "TREC"
    # TODO: for different dataset, the original data will have different format
    # TODO: this data format is related to output. Leave for future work
    # self.feature = None
    # # Convert each of the features to one-hot representation: (n_word, n_feature)
    # self.target = None
    # # Convert each of the targets to one-hot representation: (n_word, n_target)
    if self.dataset_type == "TREC":
      self.data = {}
      self.head_channel = {}
      self.head = {}
      self.head["words"] = head_words
      self.head["tags"] = head_tags
      self.sent = {}
      self.sent["words"] = sent[2:]
      self.sent["targets"] = sent[0]
      self.data["tags"] = word_tags
    else:
      self.data = {}
      self.head_channel = {}
      self.head ={}
      self.head["words"] = head_words
      self.head["tags"] = head_tags
      self.sent = {}
      self.sent["words"] = sent[1:]
      self.sent["targets"] = sent[0]
      self.sent["tags"] = word_tags

  def one_hot(self, dep_tag, pos_tag):
    # configure these to be set automatically
    basic_dependency_size = 40
    pos_size = 43
    dep_narray = np.array(dep_tag)
    pos_narray = np.array(pos_tag)

    dep_one_hot = np.zeros((dep_narray.size, basic_dependency_size + 1))
    pos_one_hot = np.zeros((pos_narray.size, pos_size + 1))

    dep_one_hot[np.arange(len(dep_tag)), dep_narray] = 1
    pos_one_hot[np.arange(len(pos_tag)), pos_narray] = 1

    concatenated_one_hot = []
    for x, y in zip(dep_one_hot, pos_one_hot):
      concatenated_one_hot.append(x.tolist() + y.tolist())

    return concatenated_one_hot

  def unfold_tags(self, tags):
    mid = len(tags) // 2
    dep_tag = tags[:mid]
    pos_tag = tags[mid:]
    return(self.one_hot(dep_tag, pos_tag))

  def convert(self, vocabs):
    words, target = vocabs
    self.data["words"] = []
    self.head_channel["words"] = []
    self.data["targets"] = target[self.sent["targets"]]

    for word in self.sent["words"]:
      self.data["words"].append(words[word])
    # for word, dep_pos in zip(self.sent["words"], self.unfold_tags(self.sent["tags"])):
    #   self.data["words"].append(words[word])
    #   # self.data["words"].extend(dep_pos)

    for word in self.head["words"]:
      self.head_channel["words"].append(words[word])
    #   self.head_channel["words"].extend(dep_pos)