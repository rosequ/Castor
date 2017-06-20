import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# from dependency_parse import get_dependency_reordered
import utils

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Trainer(object):

    def __init__(self, model, eta, mom, no_loss_reg, vec_dim, cuda=False):
        # set the random seeds for every instance of trainer.
        # needed to ensure reproduction of random word vectors for out of vocab terms
        torch.manual_seed(1234)
        np.random.seed(1234)
        self.cuda = cuda
        self.unk_term = np.random.uniform(-0.25, 0.25, vec_dim)

        self.reg = 1e-5
        self.no_loss_reg = no_loss_reg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=eta, momentum=mom, \
            weight_decay=(0 if no_loss_reg else self.reg))

        self.data_splits = {}
        self.embeddings = {}
        self.vec_dim = vec_dim

    def load_input_data(self, dataset_root_folder, word_vectors_cache_file, \
            train_set_folder, dev_set_folder, test_set_folder):
        for set_folder in [test_set_folder, dev_set_folder, train_set_folder]:
            if set_folder:
                questions, sentences, labels, maxlen_q, maxlen_s, vocab, qdeps, adeps = \
                    utils.read_in_dataset(dataset_root_folder, set_folder)


                self.data_splits[set_folder] = [questions, sentences, labels, maxlen_q, maxlen_s, qdeps, adeps]

                default_ext_feats = [np.zeros(4)] * len(self.data_splits[set_folder][0])
                self.data_splits[set_folder].append(default_ext_feats)

                utils.load_cached_embeddings(word_vectors_cache_file, vocab, self.embeddings,
                                             [] if "train" in set_folder else self.unk_term)


    def loss_regularization(self, loss):

        flattened_params = []

        for p in self.model.parameters():
            f = p.data.clone()
            flattened_params.append(f.view(-1))

        fp = torch.cat(flattened_params)

        return loss + 0.5 * self.reg * fp.norm() * fp.norm()


    def _train(self, x_q, x_a, ext_feats, x_qdeps, x_adeps, ys):
        self.optimizer.zero_grad()
        output = self.model(x_q, x_a, ext_feats, x_qdeps, x_adeps)
        loss = self.criterion(output, ys)
        if not self.no_loss_reg:
            loss.add_(self.loss_regularization(loss))
        loss.backward()
        self.optimizer.step()
        return loss.data[0], self.pred_equals_y(output, ys)

    def pred_equals_y(self, pred, y):
        _, best = pred.max(1)
        best = best.data.long().squeeze()
        return torch.sum(y.data.long() == best)

    def test(self, set_folder, batch_size):
        logger.info('----- Predictions on {} '.format(set_folder))

        questions, sentences, labels, maxlen_q, maxlen_s, qdeps, adeps, ext_feats = \
            self.data_splits[set_folder]
        word_vectors, vec_dim = self.embeddings, self.vec_dim

        self.model.eval()

        batch_size = 1

        total_loss = 0.0
        total_correct = 0.0
        
        y_pred = np.zeros(len(questions))
        ypc = 0

        for k in range(len(questions)):
            # convert raw questions and sentences to tensors
            x_q = self.get_tensorized_input_embeddings_matrix(questions[k], word_vectors, vec_dim)
            x_a = self.get_tensorized_input_embeddings_matrix(sentences[k], word_vectors, vec_dim)
            ys = Variable(torch.LongTensor([labels[k]]))
            x_qdeps = self.get_tensorized_input_embeddings_matrix(qdeps[k], word_vectors, vec_dim)
            x_adeps = self.get_tensorized_input_embeddings_matrix(adeps[k], word_vectors, vec_dim)


            pred = self.model(x_q, x_a, x_qdeps, x_adeps)
            loss = self.criterion(pred, ys)
            pred = torch.exp(pred)
            total_loss += loss
            
            y_pred[ypc] = pred.data.squeeze()[1]
            # ^ we want to score for relevance, NOT the predicted class
            ypc += 1

        logger.info('{} total {}'.format(set_folder, len(labels)))
        return y_pred


    def train(self, set_folder, batch_size, debug_single_batch):
        train_start_time = time.time()

        questions, sentences, labels, maxlen_q, maxlen_s, qdeps, adeps, ext_feats = \
            self.data_splits[set_folder]
        word_vectors, vec_dim = self.embeddings, self.vec_dim

        # set model for training
        self.model.train()

        train_loss, train_correct = 0., 0.
        for k in tqdm(range(len(questions))):

            x_q = self.get_tensorized_input_embeddings_matrix(questions[k], word_vectors, vec_dim)
            x_a = self.get_tensorized_input_embeddings_matrix(sentences[k], word_vectors, vec_dim)
            ys = Variable(torch.LongTensor([labels[k]]))
            x_qdeps = self.get_tensorized_input_embeddings_matrix(qdeps[k], word_vectors, vec_dim)
            x_adeps = self.get_tensorized_input_embeddings_matrix(adeps[k], word_vectors, vec_dim)

            ext_feats = torch.FloatTensor(ext_feats)
            ext_feats = Variable(ext_feats)
            x_ext_feats = torch.unsqueeze(ext_feats, 0)

            batch_loss, batch_correct = self._train(x_q, x_a, x_ext_feats, x_qdeps, x_adeps, ys)

            train_loss += batch_loss
            train_correct += batch_correct
            if debug_single_batch:
                break
            
        logger.info('train_loss {}'.format(train_loss))
        logger.info('train_correct {}'.format(train_correct))        
        logger.info('train_loss = {:.4f}'.format(
            train_loss/len(questions)
        ))
        logger.info('training time = {:.3f} seconds'.format(time.time() - train_start_time))
        return train_correct/len(questions)

    def get_tensorized_input_embeddings_matrix(self, sentence, word_vectors, vec_dim):
        terms = sentence.strip().split()[:60]
        word_embeddings = torch.zeros(len(terms), vec_dim).type(torch.DoubleTensor)
        num_terms = len(terms)
        for i in range(num_terms):
            word = terms[i]
            emb = torch.from_numpy(word_vectors[word]) \
                if word in word_vectors else torch.from_numpy(self.unk_term)
            word_embeddings[i] = emb

        if num_terms == 0:
            word_embeddings = torch.zeros(1, vec_dim).type(torch.DoubleTensor)
            word_embeddings[0] = torch.from_numpy(self.unk_term)

        input_tensor = torch.zeros(1, vec_dim, num_terms)
        input_tensor[0] = torch.transpose(word_embeddings, 0, 1)        
        return Variable(input_tensor)

