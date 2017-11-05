import numpy as np
import random
from time import sleep

import torch
from torchtext import data
from GPUtil import getAvailable

from args import get_args
from trec_dataset import TrecDataset
from wiki_dataset import WikiDataset
from evaluate import evaluate

args = get_args()
config = args

# Set random seed for reproducibility
torch.manual_seed(args.seed)
gpu_device = None
if not args.cuda:
    gpu_device = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    while not getAvailable(order = 'first', limit = 3, maxLoad = 0.7, maxMemory = 0.7):
        print("All devices are occupied. Waiting...")
        sleep(5)
    gpu_device = getAvailable(order = 'first', limit = 3, maxLoad = 0.7, maxMemory = 0.7)[0].item()
    torch.cuda.set_device(gpu_device)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("You have Cuda but you're using CPU for training.")
np.random.seed(args.seed)
random.seed(args.seed)

QID = data.Field(sequential=False)
QUESTION = data.Field(batch_first=True)
ANSWER = data.Field(batch_first=True)
LABEL = data.Field(sequential=False)
EXTERNAL = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False,
            postprocessing=data.Pipeline(lambda arr, _, train: [float(y) for y in arr]))
QUESTION_POS = data.Field(batch_first=True)
QUESTION_DEP = data.Field(batch_first=True)
ANSWER_POS = data.Field(batch_first=True)
ANSWER_DEP = data.Field(batch_first=True)
IDF_QUESTION = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False, pad_token="0",
            postprocessing=data.Pipeline(lambda arr, _, train: [float(x) for x in arr]))
IDF_ANSWER = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False, pad_token="0",
            postprocessing=data.Pipeline(lambda arr, _, train: [float(x) for x in arr]))
QUESTION_NUM = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False, pad_token="0",
            postprocessing=data.Pipeline(lambda arr, _, train: [float(x) for x in arr]))
ANSWER_NUM = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False, pad_token="0",
            postprocessing=data.Pipeline(lambda arr, _, train: [float(x) for x in arr]))

if config.dataset == 'TREC':
    train, dev, test = TrecDataset.splits(QID, QUESTION, QUESTION_POS, QUESTION_DEP, QUESTION, QUESTION_POS,
                                          QUESTION_DEP, ANSWER, ANSWER_POS, ANSWER_DEP, ANSWER, ANSWER_POS,
                                          ANSWER_DEP, EXTERNAL, LABEL, IDF_QUESTION, IDF_ANSWER, QUESTION_NUM,
                                          ANSWER_NUM)
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

train_iter = data.Iterator(train, batch_size=args.batch_size, device=gpu_device, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=gpu_device, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=gpu_device, train=False, repeat=False,
                                   sort=False, shuffle=False)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=gpu_device, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=gpu_device, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=gpu_device, train=False, repeat=False,
                                   sort=False, shuffle=False)

config.target_class = len(LABEL.vocab)
config.questions_num = len(QUESTION.vocab)
config.answers_num = len(ANSWER.vocab)
config.q_pos_vocab = len(QUESTION_POS.vocab)
config.q_dep_vocab = len(QUESTION_DEP.vocab)
config.a_pos_vocab = len(ANSWER_POS.vocab)
config.a_dep_vocab = len(ANSWER_DEP.vocab)

print("Dataset {}    Mode {}".format(args.dataset, args.mode))
print("VOCAB num", len(QUESTION.vocab))
print("LABEL.target_class:", len(LABEL.vocab))
print("LABELS:", LABEL.vocab.itos)
print("Train instance", len(train))
print("Dev instance", len(dev))
print("Test instance", len(test))

if args.cuda:
    model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(gpu_device))
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
