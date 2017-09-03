#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import torch.nn.functional as F


class CNNText(nn.Module):
  """
  Model class for the computational graph
  """
  def __init__(self, args):
    super(CNNText, self).__init__()


    #input_channel = args['input_channels']
    output_channel = args['output_channels']
    target_class = args['target_class']
    words_num = args['words_num']
    words_dim = args['words_dim']
    embeds_num = args['embeds_num']
    embeds_dim = args['embeds_dim']
    Ks = args['kernel_sizes']
    self.mode = args['mode']
    if self.mode == 'linguistic_multichannel':
      input_channel = 4
    elif self.mode == 'multichannel' or self.mode == 'linguistic_static':
      input_channel = 2
    else:
      input_channel = 1
    self.use_gpu = args['use_gpu']
    self.embed = nn.Embedding(words_num, words_dim)
    self.static_embed = nn.Embedding(embeds_num, embeds_dim)
    self.static_embed.weight.data.copy_(torch.from_numpy(args['embeds']))
    self.non_static_embed = nn.Embedding(embeds_num, embeds_dim)
    self.non_static_embed.weight.data.copy_(torch.from_numpy(args['embeds']))
    self.static_embed.weight.requires_grad = False

    self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2, 0))
    self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3, 0))
    self.conv3 = nn.Conv2d(input_channel, output_channel, (7, words_dim), padding=(6, 0))
    #self.convs1 = [nn.Conv2d(input_channel, output_channel, (K, words_dim), padding=(K-1, 0)) for K in Ks]

    self.dropout = nn.Dropout(args['dropout'])
    self.fc1 = nn.Linear(len(Ks) * output_channel, target_class)

  def conv_and_pool(self, x, conv):
    x = F.relu(conv(x)).squeeze(3) # (batch_size, output_channel, feature_map_dim)
    x = F.max_pool1d(x, x.size(2)).squeeze(2)
    return x

  def forward(self, head, headtag, x, wordtag):
    #if self.use_gpu:
    #  self.conv1s = [model.cuda() for model in self.convs1]
    if self.mode == 'rand':
      words = x[:,:,0]
      word_input = self.embed(words) # (batch, sent_len, embed_dim)
      x = word_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
    elif self.mode == 'static':
      static_words = x[:,:,1]
      static_input = self.static_embed(static_words)
      x = static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
    elif self.mode == 'non-static':
      non_static_words = x[:, :, 1]
      non_static_input = self.non_static_embed(non_static_words)
      x = non_static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
    elif self.mode == 'multichannel':
      words = x[:, :, 1]
      word_input = self.non_static_embed(words)  # (batch, sent_len, embed_dim)
      static_words = x[:, :, 1]
      static_input = self.static_embed(static_words)
      x = torch.stack([word_input, static_input], dim=1)# (batch, channel_input, sent_len, embed_dim)
    elif self.mode == 'linguistic_static':
      words = x[:, :, 1]
      word_channel = self.static_embed(words)  # (batch, sent_len, embed_dim)
      word_channel = torch.cat([word_channel, wordtag], 1)
      head_words = head[:, :, 1]
      headword_channel = self.static_embed(head_words)
      headword_channel = torch.cat([headword_channel, headtag], 1)
      x = torch.stack([word_channel, headword_channel], dim=1) # (batch, channel_input, sent_len, embed_dim)
    elif self.mode == 'linguistic_nonstatic':
      words = x[:, :, 1]
      word_channel = self.non_static_embed(words)  # (batch, sent_len, embed_dim)
      print(word_channel.size(), wordtag.size())
      # word_channel = torch.cat([word_channel, wordtag.unsqueeze(0).expand(word_channel.size(0), *wordtag.size())], 2)
      # print(word_channel.size())
      print(torch.type(word_channel ))
      word_channel = torch.cat((word_channel, wordtag), 0)
      head_words = head[:, :, 1]
      headword_channel = self.non_static_embed(head_words)
      headword_channel = torch.cat([headword_channel, headtag], 1)
      x = torch.stack([word_channel, headword_channel], dim=1) # (batch, channel_input, sent_len, embed_dim)
    elif self.mode == 'linguistic_multichannel':
      words = x[:, :, 1]
      word_channel_dynamic = self.non_static_embed(words)  # (batch, sent_len, embed_dim)
      word_channel_dynamic = torch.cat([word_channel_dynamic, wordtag], 1)

      word_channel_static = self.static_embed(words)  # (batch, sent_len, embed_dim)
      word_channel_static = torch.cat([word_channel_static, wordtag], 1)

      head_words = head[:, :, 1]
      headword_channel_dynamic = self.non_static_embed(head_words)
      headword_channel_dynamic = torch.cat([headword_channel_dynamic, headtag], 1)

      headword_channel_static = self.static_embed(head_words)
      headword_channel_static = torch.cat([headword_channel_static, headtag], 1)
      x = torch.stack([word_channel_dynamic, word_channel_static, headword_channel_dynamic,
                       headword_channel_static], dim=1) # (batch, channel_input, sent_len, embed_dim)
    else:
      print("Unsupported Mode")
      exit()
    #x = word_input.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
    #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
    x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
    # (batch, channel_output, ~=(sent_len)) * len(Ks)
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # max-over-time pooling
    # (batch, channel_output) * len(Ks)
    x = torch.cat(x, 1) # (batch, channel_output * len(Ks))
    x = self.dropout(x)
    logit = self.fc1(x) # (batch, target_size)
    return logit

