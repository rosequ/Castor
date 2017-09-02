#!/usr/bin/env python
# -*- coding: utf-8 -*-


from configurable import Configurable
import numpy as np


class Bucket(Configurable):
  """
  """
  def __init__(self, *args, **kwargs):
    """

    :param args:
    :param kwargs:
    """
    super(Bucket, self).__init__(*args, **kwargs)
    self._size = None
    self._data = None
    self._head = None
    self._wordtags = None
    self._headtags = None
    self._sents = None
    self._target = None

  def set_size(self, size):
    self._size = size
    self._data = []
    self._wordtags = []
    self._headtags = []
    self._sents = []
    self._head = []
    self._target = []

  def add(self, example):
    # TODO: After finalize, we can not add data anymore
    if example.length > self._size: #and self._size != -1:
      #  TODO: we may support size = -1 in the future
      raise ValueError("Bucket of size %d received sequence of len %d" % (self._size, example.length))
    self._data.append(example.data['words'])
    self._head.append(example.head_channel['words'])
    self._wordtags.append(example.data['tags'])
    self._headtags.append(example.head_channel['tags'])
    self._sents.append(example.sent['words'])
    self._target.append(example.data['targets'])
    return len(self._data)-1

  def pad(self, list, size, padding):
    return list + [padding] * abs((len(list) - size))

  def finalize(self):
    if self._data is None:
      raise ValueError("You need to set size before finalize it")

    if len(self._data) > 0:
      shape = (len(self._data), self._size, len(self._data[-1][-1]))
      data = np.zeros(shape, dtype=np.int64)
      for i, datum in enumerate(self._data):
        try:
          datum = np.array(datum)
          data[i,0:len(datum)] = datum
        except:
          print("sentence %d has Error with data :"%(i+1))
          print(datum)
          exit()
      self._data = data

      if len(self._head) > 0:
        shape = (len(self._head), self._size, len(self._head[-1][-1]))
        head = np.zeros(shape, dtype=np.int64)
        for i, datum in enumerate(self._head):
          try:
            datum = np.array(datum)
            head[i, 0:len(datum)] = datum
          except:
            print("Head: sentence %d has Error with data :" % (i + 1))
            print(self._head[i + 1])
            print(datum)
            exit()
      self._head = head

      len_max_wordtags = len(max(self._wordtags, key=len))
      wordtags = []
      for i, datum in enumerate(self._wordtags):
        wordtags.append(self.pad(datum[-1], len_max_wordtags, 0))
      self._wordtags = wordtags

      len_max_headtags = len(max(self._wordtags, key=len))
      headtags = []
      for i, datum in enumerate(self._wordtags):
        headtags.append(self.pad(datum, len_max_headtags, 0))
      self._headtags = headtags

      self._sents = np.array(self._sents)
      self._target = np.array(self._target)


    else:
      print("Finalize Error in bucket")
      exit()
    print("Bucket %s is %d x %d" % ((self._name,) + self._data.shape[0:2]))


  def __len__(self):
    return len(self._data)

  @property
  def size(self):
    return self._size
  @property
  def data(self):
    return self._data
  @property
  def wordtag(self):
    return self._wordtags
  @property
  def headtag(self):
    return self._headtags
  @property
  def head(self):
    return self._head
  @property
  def sents(self):
    return self._sents
  @property
  def target(self):
    return self._target








