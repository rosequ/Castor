import time
import os
import numpy as np
import random

import torch
import torch.nn as nn
from torchtext import data

from args import get_args
from model import SmPlusPlus
from trec_dataset import TrecDataset
from wiki_dataset import WikiDataset
from evaluate import evaluate

args = get_args()
config = args

torch.manual_seed(args.seed)

def set_vectors(field, vector_path):
    if os.path.isfile(vector_path):
        stoi, vectors, dim = torch.load(vector_path)
        field.vocab.vectors = torch.Tensor(len(field.vocab), dim)

        for i, token in enumerate(field.vocab.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                field.vocab.vectors[i] = vectors[wv_index]
            else:
                # initialize <unk> with U(-0.25, 0.25) vectors
                field.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
    else:
        print("Error: Need word embedding pt file")
        exit(1)
    return field


def regularize_loss(model, loss):
    flattened_params = []
    reg = args.weight_decay

    for p in model.parameters():
        f = p.data.clone()
        flattened_params.append(f.view(-1))

    fp = torch.cat(flattened_params)
    loss = loss + 0.5 * reg * fp.norm() * fp.norm()
    return loss

# Set default configuration in : args.py
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

QUESTION = set_vectors(QUESTION, args.vector_cache)
ANSWER = set_vectors(ANSWER, args.vector_cache)
QUESTION_POS = set_vectors(QUESTION_POS, args.pos_cache)
QUESTION_DEP = set_vectors(QUESTION_DEP, args.dep_cache)
ANSWER_POS = set_vectors(ANSWER_POS, args.pos_cache)
ANSWER_DEP = set_vectors(ANSWER_DEP, args.dep_cache)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
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

if args.resume_snapshot:
    if args.cuda:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = SmPlusPlus(config)
    model.static_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.nonstatic_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.static_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)
    model.nonstatic_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)
    model.static_q_pos_embed.weight.data.copy_(QUESTION_POS.vocab.vectors)
    model.nonstatic_q_pos_embed.weight.data.copy_(QUESTION_POS.vocab.vectors)
    model.static_a_pos_embed.weight.data.copy_(ANSWER_POS.vocab.vectors)
    model.nonstatic_a_pos_embed.weight.data.copy_(ANSWER_POS.vocab.vectors)
    model.static_q_dep_embed.weight.data.copy_(QUESTION_DEP.vocab.vectors)
    model.nonstatic_q_dep_embed.weight.data.copy_(QUESTION_DEP.vocab.vectors)
    model.static_a_dep_embed.weight.data.copy_(ANSWER_DEP.vocab.vectors)
    model.nonstatic_a_dep_embed.weight.data.copy_(ANSWER_DEP.vocab.vectors)

    if args.cuda:
        model.cuda()
        print("Shift model to GPU")


parameter = filter(lambda p: p.requires_grad, model.parameters())

# the SM model originally follows SGD but Adadelta is used here
optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
early_stop = False
best_dev_map = 0
iterations = 0
iters_not_improved = 0
epoch = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
print(header)

index2label = np.array(LABEL.vocab.itos)
index2qid = np.array(QID.vocab.itos)
index2answer = np.array(ANSWER.vocab.itos)

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, best_dev_map))
        break
    epoch += 1
    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        model.train(); optimizer.zero_grad()
        try:
            scores = model(batch)
        except RuntimeError as e:
            # for qid, tensor in zip(index2qid[batch.qid.cpu().data.numpy()], index2answer[batch.answer.cpu().data.numpy()]):
                # print(qid, tensor)
            print(e)
        n_correct += (torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(scores, batch.label)
        loss = regularize_loss(model, loss)
        loss.backward()
        optimizer.step()

        # Evaluate performance on validation set
        if iterations % args.dev_every == 1:
            # switch model into evaluation mode
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct = 0
            dev_losses = []
            instance = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                qid_array = index2qid[np.transpose(dev_batch.qid.cpu().data.numpy())]
                true_label_array = index2label[np.transpose(dev_batch.label.cpu().data.numpy())]

                scores = model(dev_batch)
                n_dev_correct += (torch.max(scores, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(scores, dev_batch.label)
                dev_losses.append(dev_loss.data[0])
                index_label = np.transpose(torch.max(scores, 1)[1].view(dev_batch.label.size()).cpu().data.numpy())
                label_array = index2label[index_label]
                # get the relevance scores
                score_array = scores[:, 2].cpu().data.numpy()
                for i in range(dev_batch.batch_size):
                    this_qid, predicted_label, score, gold_label = qid_array[i], label_array[i], score_array[i], true_label_array[i]
                    instance.append((this_qid, predicted_label, score, gold_label))


            dev_map, dev_mrr = evaluate(instance, config.dataset, 'valid', config.mode)
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0],
                                          sum(dev_losses) / len(dev_losses), train_acc, dev_map))

            # Update validation results
            if dev_map > best_dev_map:
                iters_not_improved = 0
                best_dev_map = dev_map
                snapshot_path = os.path.join(args.save_path, args.dataset, args.mode+'_best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                                      n_correct / n_total * 100, ' ' * 12))
