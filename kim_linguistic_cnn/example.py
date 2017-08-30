#!/usr/bin/env python
# -*- coding: utf-8 -*-



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
      self.sent["tags"] = word_tags
    else:
      self.data = {}
      self.head_channel = {}
      self.head ={}
      self.head["words"] = head_words
      self.head["tags"] = head_tags
      self.sent = {}
      self.sent["words"] = sent[1:]
      self.sent["targets"] = sent[0]
      self.sent["ags"] = word_tags

  def convert(self, vocabs):
    words, target = vocabs
    self.data["words"] = []
    self.data["targets"] = target[self.sent["targets"]]
    for word in self.sent["words"]:
      self.data["words"].append(words[word])

    self.data["words"].extend(self.sent["tags"])

    for word in self.head["words"]:
      self.head_channel["words"].append(words[word])
    self.head_channel["words"].extend(self.head["tags"])