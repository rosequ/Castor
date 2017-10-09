import torch
import torch.nn as nn

import torch.nn.functional as F

class SmPlusPlus(nn.Module):
    def __init__(self, config):
        super(SmPlusPlus, self).__init__()
        output_channel = config.output_channel
        questions_num = config.questions_num
        answers_num = config.answers_num
        words_dim = config.words_dim
        filter_width = config.filter_width
        q_pos_size = config.q_pos_vocab
        q_dep_size = config.q_dep_vocab
        a_pos_size = config.a_pos_vocab
        a_dep_size = config.a_dep_vocab
        self.mode = config.mode
        pos_size = 44
        dep_size = 41
        conv_dim = words_dim

        n_classes = config.target_class
        ext_feats_size = 4

        if self.mode == 'linguistic_multichannel':
            input_channel = 4
        elif self.mode == 'multichannel' or self.mode == 'linguistic_static' or self.mode == 'linguistic_nonstatic':
            input_channel = 2
        else:
            input_channel = 1

        if 'linguistic_nonstatic' in self.mode or 'linguistic_static' in self.mode:
            conv_dim += (pos_size + dep_size)
        elif 'linguistic_head' in self.mode:
            conv_dim += (pos_size + dep_size + words_dim)

        self.question_embed = nn.Embedding(questions_num, words_dim)
        self.answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed = nn.Embedding(questions_num, words_dim)
        self.nonstatic_question_embed = nn.Embedding(questions_num, words_dim)
        self.static_answer_embed = nn.Embedding(answers_num, words_dim)
        self.nonstatic_answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed.weight.requires_grad = False
        self.static_answer_embed.weight.requires_grad = False

        self.static_q_pos_embed = nn.Embedding(q_pos_size, pos_size)
        self.static_a_pos_embed = nn.Embedding(a_pos_size, pos_size)
        self.nonstatic_q_pos_embed = nn.Embedding(q_pos_size, pos_size)
        self.nonstatic_a_pos_embed = nn.Embedding(a_pos_size, pos_size)
        self.static_q_pos_embed.weight.requires_grad = False
        self.static_a_pos_embed.weight.requires_grad = False

        self.static_q_dep_embed = nn.Embedding(q_dep_size, dep_size)
        self.static_a_dep_embed = nn.Embedding(a_dep_size, dep_size)
        self.nonstatic_q_dep_embed = nn.Embedding(q_dep_size, dep_size)
        self.nonstatic_a_dep_embed = nn.Embedding(a_dep_size, dep_size)
        self.static_q_dep_embed.weight.requires_grad = False
        self.static_a_dep_embed.weight.requires_grad = False

        self.conv_q = nn.Conv2d(input_channel, output_channel, (filter_width, conv_dim), padding=(filter_width - 1, 0))
        self.conv_a = nn.Conv2d(input_channel, output_channel, (filter_width, conv_dim), padding=(filter_width - 1, 0))

        self.dropout = nn.Dropout(config.dropout)
        n_hidden = 2 * output_channel + ext_feats_size

        self.combined_feature_vector = nn.Linear(n_hidden, n_hidden)
        self.hidden = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x_question = x.question
        x_q_pos = x.question_word_pos
        x_q_dep = x.question_word_dep
        x_answer = x.answer
        x_a_pos = x.answer_word_pos
        x_a_dep = x.answer_word_dep
        x_ext = x.ext_feat
        head_q = x.head_question
        head_q_pos = x.head_q_pos
        head_q_dep = x.head_q_dep
        head_a = x.head_answer
        head_a_pos = x.head_a_pos
        head_a_dep = x.head_a_dep

        if self.mode == 'rand':
            question = self.question_embed(x_question).unsqueeze(1)
            answer = self.answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # actual SM model mode (Severyn & Moschitti, 2015)
        elif self.mode == 'static':
            question = self.static_question_embed(x_question).unsqueeze(1)
            answer = self.static_answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'non-static':
            question = self.nonstatic_question_embed(x_question).unsqueeze(1)
            answer = self.nonstatic_answer_embed(x_answer).unsqueeze(1) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'multichannel':
            question_static = self.static_question_embed(x_question)
            answer_static = self.static_answer_embed(x_answer)
            question_nonstatic = self.nonstatic_question_embed(x_question)
            answer_nonstatic = self.nonstatic_answer_embed(x_answer)
            question = torch.stack([question_static, question_nonstatic], dim=1)
            answer = torch.stack([answer_static, answer_nonstatic], dim=1)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'linguistic_static':
            x_question = self.static_question_embed(x_question)
            x_q_pos = self.static_q_pos_embed(x_q_pos)
            x_q_dep = self.static_q_dep_embed(x_q_dep)
            q_word_channel = torch.cat([x_question, x_q_pos, x_q_dep], 2)
            head_question = self.static_question_embed(head_q)
            head_q_pos = self.static_q_pos_embed(head_q_pos)
            head_q_dep = self.static_q_dep_embed(head_q_dep)
            q_head_channel = torch.cat([head_question, head_q_pos, head_q_dep], 2)
            question = torch.stack([q_head_channel, q_word_channel], dim=1)

            x_answer = self.static_answer_embed(x_answer)
            x_a_pos = self.static_a_pos_embed(x_a_pos)
            x_a_dep = self.static_a_dep_embed(x_a_dep)
            a_word_channel = torch.cat([x_answer, x_a_pos, x_a_dep], 2)
            head_a = self.static_answer_embed(head_a)
            head_a_pos = self.static_a_pos_embed(head_a_pos)
            head_a_dep = self.static_a_dep_embed(head_a_dep)
            a_head_channel = torch.cat([head_a, head_a_pos, head_a_dep], 2)
            answer = torch.stack([a_head_channel, a_word_channel], dim=1)

            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'linguistic_nonstatic':
            x_question = self.nonstatic_question_embed(x_question)
            x_q_pos = self.nonstatic_q_pos_embed(x_q_pos)
            x_q_dep = self.nonstatic_q_dep_embed(x_q_dep)
            q_word_channel = torch.cat([x_question, x_q_pos, x_q_dep], 2)
            head_question = self.nonstatic_question_embed(head_q)
            head_q_pos = self.nonstatic_q_pos_embed(head_q_pos)
            head_q_dep = self.nonstatic_q_dep_embed(head_q_dep)
            q_head_channel = torch.cat([head_question, head_q_pos, head_q_dep], 2)
            question = torch.stack([q_head_channel, q_word_channel], dim=1)

            x_answer = self.nonstatic_answer_embed(x_answer)
            x_a_pos = self.nonstatic_a_pos_embed(x_a_pos)
            x_a_dep = self.nonstatic_a_dep_embed(x_a_dep)
            a_word_channel = torch.cat([x_answer, x_a_pos, x_a_dep], 2)
            head_a = self.nonstatic_answer_embed(head_a)
            head_a_pos = self.nonstatic_a_pos_embed(head_a_pos)
            head_a_dep = self.nonstatic_a_dep_embed(head_a_dep)
            a_head_channel = torch.cat([head_a, head_a_pos, head_a_dep], 2)
            answer = torch.stack([a_head_channel, a_word_channel], dim=1)

            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling

        elif self.mode == 'linguistic_multichannel':
            x_question_static = self.static_question_embed(x_question)
            x_q_pos_static = self.static_q_pos_embed(x_q_pos)
            x_q_dep_static = self.static_q_dep_embed(x_q_dep)

            x_question_dynamic = self.nonstatic_question_embed(x_question)
            x_q_pos_dynamic = self.nonstatic_q_pos_embed(x_q_pos)
            x_q_dep_dynamic = self.nonstatic_q_dep_embed(x_q_dep)

            head_question_static = self.static_question_embed(head_q)
            head_q_pos_static = self.static_q_pos_embed(head_q_pos)
            head_q_dep_static = self.static_q_dep_embed(head_q_dep)

            head_question_dynamic = self.nonstatic_question_embed(head_q)
            head_q_pos_dynamic = self.nonstatic_q_pos_embed(head_q_pos)
            head_q_dep_dynamic = self.nonstatic_q_dep_embed(head_q_dep)

            q_word_channel_static = torch.cat([x_question_static, x_q_pos_static, x_q_dep_static], 2)
            q_word_channel_dynamic = torch.cat([x_question_dynamic, x_q_pos_dynamic, x_q_dep_dynamic], 2)
            q_head_channel_static = torch.cat([head_question_static, head_q_pos_static, head_q_dep_static], 2)
            q_head_channel_dynamic = torch.cat([head_question_dynamic, head_q_pos_dynamic, head_q_dep_dynamic], 2)
            question_channel = torch.stack([q_word_channel_static, q_word_channel_dynamic, q_head_channel_static,
                                            q_head_channel_dynamic], dim=1)

            x_answer_static = self.static_question_embed(x_answer)
            x_a_pos_static = self.static_a_pos_embed(x_a_pos)
            x_a_dep_static = self.static_a_dep_embed(x_a_dep)

            x_answer_dynamic = self.nonstatic_question_embed(x_answer)
            x_a_pos_dynamic = self.nonstatic_a_pos_embed(x_a_pos)
            x_a_dep_dynamic = self.nonstatic_a_dep_embed(x_a_dep)

            head_answer_static = self.static_question_embed(head_a)
            head_a_pos_static = self.static_a_pos_embed(head_a_pos)
            head_a_dep_static = self.static_a_dep_embed(head_a_dep)

            head_answer_dynamic = self.nonstatic_question_embed(head_a)
            head_a_pos_dynamic = self.nonstatic_a_pos_embed(head_a_pos)
            head_a_dep_dynamic = self.nonstatic_a_dep_embed(head_a_dep)

            a_word_channel_static = torch.cat([x_answer_static, x_a_pos_static, x_a_dep_static], 2)
            a_word_channel_dynamic = torch.cat([x_answer_dynamic, x_a_pos_dynamic, x_a_dep_dynamic], 2)
            a_head_channel_static = torch.cat([head_answer_static, head_a_pos_static, head_a_dep_static], 2)
            a_head_channel_dynamic = torch.cat([head_answer_dynamic, head_a_pos_dynamic, head_a_dep_dynamic], 2)
            answer_channel = torch.stack([a_word_channel_static, a_word_channel_dynamic, a_head_channel_static,
                                            a_head_channel_dynamic], dim=1)

            x = [F.tanh(self.conv_q(question_channel)).squeeze(3), F.tanh(self.conv_a(answer_channel)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        else:
            print("Unsupported Mode")
            exit()

        # append external features and feed to fc
        x.append(x_ext)
        x = torch.cat(x, 1)

        x = F.tanh(self.combined_feature_vector(x))
        x = self.dropout(x)
        x = self.hidden(x)
        return x