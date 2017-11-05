import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable

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
        pos_dim = config.pos_dim
        dep_dim = config.dep_dim
        conv_dim = words_dim

        n_classes = config.target_class
        ext_feats_size = 4

        if self.mode == 'ling_sep_dep':
            input_channel = 3
        else:
            input_channel = 1


        print(conv_dim, input_channel)
        self.question_embed = nn.Embedding(questions_num, words_dim)
        self.answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed = nn.Embedding(questions_num, words_dim)
        self.nonstatic_question_embed = nn.Embedding(questions_num, words_dim)
        self.static_answer_embed = nn.Embedding(answers_num, words_dim)
        self.nonstatic_answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed.weight.requires_grad = False
        self.static_answer_embed.weight.requires_grad = False

        print(pos_dim)
        # nn.Embedding(q_pos_size, pos_dim)
        self.static_q_pos_embed = nn.Embedding(q_pos_size, pos_dim)
        self.static_a_pos_embed = nn.Embedding(a_pos_size, pos_dim)
        self.nonstatic_q_pos_embed = nn.Embedding(q_pos_size, pos_dim)
        self.nonstatic_a_pos_embed = nn.Embedding(a_pos_size, pos_dim)
        self.static_q_pos_embed.weight.requires_grad = False
        self.static_a_pos_embed.weight.requires_grad = False

        self.static_q_dep_embed = nn.Embedding(q_dep_size, dep_dim)
        self.static_a_dep_embed = nn.Embedding(a_dep_size, dep_dim)
        self.nonstatic_q_dep_embed = nn.Embedding(q_dep_size, dep_dim)
        self.nonstatic_a_dep_embed = nn.Embedding(a_dep_size, dep_dim)
        self.static_q_dep_embed.weight.requires_grad = False
        self.static_a_dep_embed.weight.requires_grad = False

        print(pos_dim, dep_dim, conv_dim, input_channel, output_channel, filter_width)
        self.conv_q = nn.Conv2d(input_channel, output_channel, (filter_width, conv_dim), padding=(filter_width - 1, 0))
        self.conv_a = nn.Conv2d(input_channel, output_channel, (filter_width, conv_dim), padding=(filter_width - 1, 0))
        self.dropout = nn.Dropout(config.dropout)
        n_hidden = 2 * output_channel + ext_feats_size

        self.combined_feature_vector = nn.Linear(n_hidden, n_hidden)
        self.hidden = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x_question = x.question
        x_q_dep = x.question_word_dep
        x_answer = x.answer
        x_a_dep = x.answer_word_dep
        x_ext = x.ext_feat
        head_q = x.head_question
        head_a = x.head_answer

        if self.mode == 'ling_sep_dep':
            x_question = self.nonstatic_question_embed(x_question)
            x_q_head = self.nonstatic_question_embed(head_q)
            x_q_dep = self.nonstatic_q_dep_embed(x_q_dep)
            # todo choose CPU variable if no_cuda
            padding_q_dep = Variable(
                torch.zeros(x_question.size(0), x_question.size(1), x_question.size(2) - x_q_dep.size(2))).cuda()
            padded_q_dep = torch.cat([x_q_dep, padding_q_dep], 2)
            question = torch.stack([x_question, x_q_head, padded_q_dep], 1)

            x_answer = self.nonstatic_answer_embed(x_answer)
            x_a_head = self.nonstatic_answer_embed(head_a)
            x_a_dep = self.nonstatic_a_dep_embed(x_a_dep)
            padding_a_dep = Variable(
                torch.zeros(x_answer.size(0), x_answer.size(1), x_answer.size(2) - x_a_dep.size(2))).cuda()
            padded_a_dep = torch.cat([x_a_dep, padding_a_dep], 2)
            answer = torch.stack([x_answer, x_a_head, padded_a_dep], 1)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
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