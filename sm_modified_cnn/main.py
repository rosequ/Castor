import numpy as np
import random
import logging

import torch
from torchtext import data

from args import get_args
from trec_dataset import TrecDataset
from wiki_dataset import WikiDataset
from evaluate import evaluate

args = get_args()
config = args

# Set random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("You have Cuda but you're using CPU for training.")
np.random.seed(args.seed)
random.seed(args.seed)

QID = data.Field(sequential=False)
QUESTION = data.Field(batch_first=True)
ANSWER = data.Field(batch_first=True)
LABEL = data.Field(sequential=False)
EXTERNAL = data.Field(sequential=False, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False,
                      preprocessing=data.Pipeline(lambda x: x.split()),
                      postprocessing=data.Pipeline(lambda x, train: [float(y) for y in x]))
QUESTION_POS = data.Field(batch_first=True)
QUESTION_DEP = data.Field(batch_first=True)
ANSWER_POS = data.Field(batch_first=True)
ANSWER_DEP = data.Field(batch_first=True)
HEAD_QUESTION = data.Field(batch_first=True)
HEAD_QUESTION_POS = data.Field(batch_first=True)
HEAD_QUESTION_DEP = data.Field(batch_first=True)
HEAD_ANSWER = data.Field(batch_first=True)
HEAD_ANSWER_POS = data.Field(batch_first=True)
HEAD_ANSWER_DEP = data.Field(batch_first=True)

if config.dataset == 'TREC':
    train, dev, test = TrecDataset.splits(QID, QUESTION, QUESTION_POS, QUESTION_DEP, HEAD_QUESTION, HEAD_QUESTION_POS,
                                          HEAD_QUESTION_DEP, ANSWER, ANSWER_POS, ANSWER_DEP, HEAD_ANSWER, HEAD_ANSWER_POS,
                                          HEAD_ANSWER_DEP, EXTERNAL, LABEL)
elif config.dataset == 'wiki':
    train, dev, test = WikiDataset.splits(QID, QUESTION, ANSWER, EXTERNAL, LABEL)
else:
    print("Unsupported dataset")
    exit()

QID.build_vocab(train, dev, test)
QUESTION.build_vocab(train, dev, test)
ANSWER.build_vocab(train, dev, test)
LABEL.build_vocab(train, dev, test)
QUESTION_POS.build_vocab(train, dev, test)
QUESTION_DEP.build_vocab(train, dev, test)
ANSWER_POS.build_vocab(train, dev, test)
ANSWER_DEP.build_vocab(train, dev, test)
HEAD_QUESTION.build_vocab(train, dev, test)
HEAD_QUESTION_POS.build_vocab(train, dev, test)
HEAD_QUESTION_DEP.build_vocab(train, dev, test)
HEAD_ANSWER.build_vocab(train, dev, test)
HEAD_ANSWER_POS.build_vocab(train, dev, test)
HEAD_ANSWER_DEP.build_vocab(train, dev, test)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

config.target_class = len(LABEL.vocab)
config.questions_num = len(QUESTION.vocab)
config.answers_num = len(ANSWER.vocab)
print("Label dict:", LABEL.vocab.itos)

if args.cuda:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = torch.load(args.trained_model, map_location=lambda storage,location: storage)

index2label = np.array(LABEL.vocab.itos)
index2qid = np.array(QID.vocab.itos)

def predict(dataset, test_mode, dataset_iter):
    model.eval()
    dataset_iter.init_epoch()

    instance = []
    for dev_batch_idx, dev_batch in enumerate(dataset_iter):
        qid_array = index2qid[np.transpose(dev_batch.qid.cpu().data.numpy())]
        true_label_array = index2label[np.transpose(dev_batch.label.cpu().data.numpy())]

        scores = model(dev_batch)

        index_label = np.transpose(torch.max(scores, 1)[1].view(dev_batch.label.size()).cpu().data.numpy())
        label_array = index2label[index_label]
        score_array = scores[:, 2].cpu().data.numpy()
        # print and write the result
        for i in range(dev_batch.batch_size):
            this_qid, predicted_label, score, gold_label = qid_array[i], label_array[i], score_array[i], \
                                                           true_label_array[i]
            instance.append((this_qid, predicted_label, score, gold_label))

    dev_map, dev_mrr = evaluate(instance, dataset, test_mode, config.mode)
    print(dev_map, dev_mrr)

# Run the model on the dev set
predict(config.dataset, 'dev', dataset_iter=dev_iter)

# Run the model on the test set
predict(config.dataset, 'test', dataset_iter=test_iter)
