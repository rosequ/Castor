import torch
import torch.nn as nn
import torch.nn.functional as F

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class QAModel(nn.Module):

    @staticmethod
    def save(model, model_fname):
        torch.save(model, model_fname)


    @staticmethod
    def load(model_fname):
        return torch.load(model_fname)


    def __init__(self, input_n_dim, filter_width, num_conv_filters=100, \
                 ext_feats_size=4, n_classes=2, cuda=False):
        super(QAModel, self).__init__()

        self.conv_channels = num_conv_filters

        # question: regular input word embeddings
        self.conv_q = nn.Sequential(
            nn.Conv1d(input_n_dim, self.conv_channels, filter_width, padding=filter_width-1),
            nn.Tanh()
        )
        # question: dependency parsing reordered word embeddings
        self.conv_q_deps = nn.Sequential(
            nn.Conv1d(input_n_dim, self.conv_channels, 2, stride=2),
            nn.Tanh()
        )
        # answer: regular input word embeddings
        self.conv_a = nn.Sequential(
            nn.Conv1d(input_n_dim, self.conv_channels, filter_width, padding=filter_width-1),
            nn.Tanh()
        )
        # answer: dependency parsing reordered word embeddings
        self.conv_a_deps = nn.Sequential(
            nn.Conv1d(input_n_dim, self.conv_channels, 2, stride=2),
            nn.Tanh()
        )

        # TODO: how to combine the outputs of the multiple perspectives
        # 1. single combined feature vector        
        num_combined_features = 4*self.conv_channels + 4
        self.feature2output = nn.Sequential(
            nn.Linear(num_combined_features, num_combined_features),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(num_combined_features, n_classes),
            nn.LogSoftmax()
        )
        # TODO 2. linear combination of 2 comparisons
        # TODO 3. something else

        if cuda and torch.cuda.is_available():
            self.conv_q, self.conv_a = self.conv_q.cuda(), self.conv_a.cuda()
            self.combined_feature_vector = self.combined_feature_vector.cuda()
            self.combined_features_activation = self.combined_features_activation.cuda()
            self.dropout, self.hidden, self.logsoftmax = self.dropout.cuda(), \
                self.hidden.cuda(), self.logsoftmax.cuda()

    def forward(self, question, answer, ext_feat, q_deps, a_deps):
        # push regular question forward
        q_reg = self.conv_q.forward(question)
        q_reg = F.max_pool1d(q_reg, q_reg.size()[2])
        q_reg = q_reg.view(-1, self.conv_channels)
        # push dependancy parsed question forward
        q_dep = self.conv_q.forward(question)
        q_dep = F.max_pool1d(q_dep, q_dep.size()[2])
        q_dep = q_dep.view(-1, self.conv_channels)
        # push regular answer forward
        a_reg = self.conv_a.forward(answer)
        a_reg = F.max_pool1d(a_reg, a_reg.size()[2])
        a_reg = a_reg.view(-1, self.conv_channels)
        # push dependancy parsed answer forward
        a_dep = self.conv_a.forward(answer)
        a_dep = F.max_pool1d(a_dep, a_dep.size()[2])
        a_dep = a_dep.view(-1, self.conv_channels)


        # TODO: combining outputs: there can be various options for this
        # make combined feature vector
        x = torch.cat([q_reg, q_dep, a_reg, a_dep, ext_feat], 1)
        x = self.feature2output.forward(x)
        return x


