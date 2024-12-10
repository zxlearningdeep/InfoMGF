import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_data
from model import GCN, GCL, AGG
from graph_learners import *
from utils import *
from params import *
from augment import *
from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans as KMeans_py
from sklearn.metrics import f1_score

import random

EOS = 1e-10
args = set_params()

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()
        self.training = False

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


    def test_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        test_accu = accuracy(logp[mask], labels[mask])

        preds = torch.argmax(logp, dim=1)
        test_f1_macro = torch.tensor(f1_score(labels[mask].cpu(), preds[mask].cpu(), average='macro'))
        test_f1_micro = torch.tensor(f1_score(labels[mask].cpu(), preds[mask].cpu(), average='micro'))

        return loss, test_accu, test_f1_macro, test_f1_micro

    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            loss, train_accu = self.loss_cls(model, train_mask, features, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, val_accu = self.loss_cls(model, val_mask, features, labels)
                if val_accu > best_val:
                    bad_counter = 0
                    best_val = val_accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break
        best_model.eval()
        test_loss, test_accu, test_f1_macro, test_f1_micro = self.test_cls(best_model, test_mask, features, labels)
        return best_val, test_accu, test_f1_macro, test_f1_micro

    def loss_discriminator(self, discriminator, model, specific_graph_learner, features, view_features, adjs, optimizer_discriminator):

        optimizer_discriminator.zero_grad()

        learned_specific_adjs = []
        for i in range(len(adjs)):
            specific_adjs_embedding = specific_graph_learner[i](view_features[i])
            learned_specific_adj = specific_graph_learner[i].graph_process(specific_adjs_embedding)
            learned_specific_adjs.append(learned_specific_adj)

        z_specific_adjs = [model(features, learned_specific_adjs[i]) for i in range(len(adjs))]

        adjs_aug = graph_generative_augment(adjs, features, discriminator, sparse=args.sparse)
        z_aug_adjs = [model(features, adjs_aug[i]) for i in range(len(adjs))]
        loss_dis = discriminator.cal_loss_dis(z_aug_adjs, z_specific_adjs, view_features)

        loss_dis.backward()
        optimizer_discriminator.step()
        return loss_dis

    def loss_gcl(self, model, specific_graph_learner, fused_graph_learner, features, view_features, adjs,
                 optimizer, discriminator=None):
        optimizer.zero_grad()

        learned_specific_adjs = []
        for i in range(len(adjs)):
            specific_adjs_embedding = specific_graph_learner[i](view_features[i])
            learned_specific_adj = specific_graph_learner[i].graph_process(specific_adjs_embedding)
            learned_specific_adjs.append(learned_specific_adj)

        fused_embedding = fused_graph_learner(torch.cat(view_features, dim=1))
        learned_fused_adj = fused_graph_learner.graph_process(fused_embedding)
        z_specific_adjs = [model(features, learned_specific_adjs[i]) for i in range(len(adjs))]
        z_fused_adj = model(features, learned_fused_adj)

        if args.augment_type == 'random':
            adjs_aug = graph_augment(adjs, args.dropedge_rate, training=self.training, sparse=args.sparse)
        elif args.augment_type == 'generative':
            adjs_aug = graph_generative_augment(adjs, features, discriminator, sparse=args.sparse)
        if args.sparse:
            for i in range(len(adjs)):
                adjs_aug[i].edata['w'] = adjs_aug[i].edata['w'].detach()
        else:
            adjs_aug = [a.detach() for a in adjs_aug]
        z_aug_adjs = [model(features, adjs_aug[i]) for i in range(len(adjs))]


        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.cal_loss([z[batch] for z in z_specific_adjs], [z[batch] for z in z_aug_adjs], z_fused_adj[batch]) * weight
        else:
            loss = model.cal_loss(z_specific_adjs, z_aug_adjs, z_fused_adj)

        loss.backward()
        optimizer.step()

        return loss

    def train(self, args):
        print(args)

        torch.cuda.set_device(args.gpu)

        features_original, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adjs_original = load_data(args)

        if args.downstream_task == 'classification':
            test_accuracies = []
            test_maf1 = []
            test_mif1 = []
            validation_accuracies = []

            fh = open("result_" + args.dataset + "_Class.txt", "a")
            print(args, file=fh)
            fh.write('\r\n')
            fh.flush()
            fh.close()

        if args.downstream_task == 'classification':
            fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
            print(args, file=fh)
            fh.write('\r\n')
            fh.flush()
            fh.close()

        for trial in range(args.ntrials):

            self.setup_seed(trial)
            adjs = copy.deepcopy(adjs_original)
            features = copy.deepcopy(features_original)
            view_features = AGG([features for _ in range(len(adjs))], adjs, args.r, sparse=args.sparse)
            view_features.append(features)

            specific_graph_learner = [ATT_learner(2, features.shape[1], args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner) for _ in range(len(adjs))]
            fused_graph_learner = ATT_learner(2, features.shape[1]*(len(adjs)+1), args.k, 6, args.dropedge_rate, args.sparse, args.activation_learner)

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, sparse=args.sparse, num_g=len(adjs))

            optimizer = torch.optim.Adam([{'params': specific_graph_learner[i].parameters()} for i in range(len(adjs))] +
                                        [{'params': fused_graph_learner.parameters()}] +
                                        [{'params': model.parameters()}], lr=args.lr, weight_decay=args.w_decay)
            if args.augment_type == 'generative':
                discriminator = Discriminator(nfeats, args.hidden_dim, args.rep_dim, args.aug_lambda, args.dropout, args.dropedge_rate)
                optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_dis, weight_decay=args.w_decay)
            else:
                discriminator = None
                optimizer_discriminator = None

            if torch.cuda.is_available():
                model = model.cuda()
                specific_graph_learner = [m.cuda() for m in specific_graph_learner]
                fused_graph_learner = fused_graph_learner.cuda()
                if args.augment_type == 'generative':
                    discriminator.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                adjs = [adj.cuda() for adj in adjs]
                view_features = [f.cuda() for f in view_features]

            for epoch in range(1, args.epochs + 1):

                model.train()
                [learner.train() for learner in specific_graph_learner]
                fused_graph_learner.train()
                if args.augment_type == 'generative':
                    discriminator.eval()
                self.training = True

                loss = self.loss_gcl(model, specific_graph_learner, fused_graph_learner, features, view_features, adjs,
                                    optimizer, discriminator)
                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

                if args.augment_type == 'generative':
                    discriminator.train()
                    model.eval()
                    [learner.eval() for learner in specific_graph_learner]
                    fused_graph_learner.eval()
                    loss_dis = self.loss_discriminator(discriminator, model, specific_graph_learner, features, view_features, adjs, optimizer_discriminator)

                    print("Epoch {:05d} | DIS Loss {:.4f}".format(epoch, loss_dis.item()))

                if epoch % args.eval_freq == 0:

                    # get learned graph
                    model.eval()
                    [learner.eval() for learner in specific_graph_learner]
                    fused_graph_learner.eval()
                    if args.augment_type == 'generative':
                        discriminator.eval()
                    self.training = False

                    fused_embedding = fused_graph_learner(torch.cat(view_features, dim=1))
                    learned_fused_adj = fused_graph_learner.graph_process(fused_embedding)

                    if args.sparse:
                        learned_fused_adj.edata['w'] = learned_fused_adj.edata['w'].detach()
                    else:
                        learned_fused_adj = learned_fused_adj.detach()

                    if args.downstream_task == 'classification':
                        f_adj = learned_fused_adj

                        val_accu, test_accu, test_f1_macro, test_f1_micro = self.evaluate_adj_by_cls(f_adj, features, nfeats, labels,
                                                                               nclasses, train_mask, val_mask, test_mask, args)
                        print('EPOCH:', epoch, ' val_acc:', val_accu, ' test_acc', test_accu, ' test maf1:', test_f1_macro, ' test mif1:', test_f1_micro)

                        fh = open("result_" + args.dataset + "_Class.txt", "a")
                        fh.write(
                            'Trial=%f, Epoch=%f, val_acc=%f, test_accu=%f, test_f1_macro=%f,  test_f1_micro=%f' % (trial, epoch, val_accu, test_accu, test_f1_macro, test_f1_micro))
                        fh.write('\r\n')
                        fh.flush()
                        fh.close()

                    if args.downstream_task == 'clustering':
                        embedding_ = model(features, learned_fused_adj)
                        embedding_ = embedding_.detach()

                        acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []
                        for clu_trial in range(10):
                            if args.sparse:
                                embedding = embedding_
                                y_pred, _ = KMeans_py(X=embedding, num_clusters=nclasses, distance='euclidean',
                                                   device='cuda')
                                predict_labels = y_pred.cpu().numpy()
                            else:
                                embedding = embedding_.cpu().numpy()
                                kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial).fit(embedding)
                                predict_labels = kmeans.predict(embedding)
                            cm_all = clustering_metrics(labels.cpu().numpy(), predict_labels)
                            acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                            acc_mr.append(acc_)
                            nmi_mr.append(nmi_)
                            f1_mr.append(f1_)
                            ari_mr.append(ari_)

                        acc, nmi, f1, ari = np.mean(acc_mr), np.mean(nmi_mr), np.mean(f1_mr), np.mean(ari_mr)
                        print("Epoch {:05d} | acc {:.4f} | f1 {:.4f} | nmi {:.4f} | ari {:.4f}".format(epoch, acc, f1, nmi, ari))

                        fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
                        fh.write(
                            'Trial=%f, Epoch=%f, ACC=%f, f1_macro=%f,  NMI=%f, ADJ_RAND_SCORE=%f' % (trial, epoch, acc, f1, nmi, ari))
                        fh.write('\r\n')
                        fh.flush()
                        fh.close()

            self.training = False
            if args.downstream_task == 'classification':
                validation_accuracies.append(val_accu.item())
                test_accuracies.append(test_accu.item())
                test_maf1.append(test_f1_macro.item())
                test_mif1.append(test_f1_micro.item())
                print("Trial: ", trial + 1)
                print("Best val ACC: ", val_accu.item())
                print("Best test ACC: ", test_accu.item())
                print("Best test MaF1: ", test_f1_macro.item())
                print("Best test MiF1: ", test_f1_micro.item())
            elif args.downstream_task == 'clustering':
                print("Final ACC: ", acc)
                print("Final NMI: ", nmi)
                print("Final F-score: ", f1)
                print("Final ARI: ", ari)

        if args.downstream_task == 'classification' and trial != 0:
            self.print_results(validation_accuracies, test_accuracies, test_maf1, test_mif1)


    def print_results(self, validation_accu, test_accu, test_maf1, test_mif1):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        maf1_test = "Test maf1: {:.4f} +/- {:.4f}".format(np.mean(test_maf1),np.std(test_maf1))
        mif1_test = "Test mif1: {:.4f} +/- {:.4f}".format(np.mean(test_mif1),np.std(test_mif1))
        print(s_val)
        print(s_test)
        print(maf1_test)
        print(mif1_test)

        fh = open("result_" + args.dataset + "_Class.txt", "a")
        fh.write("Test maf1: {:.4f} +/- {:.4f}".format(np.mean(test_maf1),np.std(test_maf1)))
        fh.write('\r\n')
        fh.write("Test mif1: {:.4f} +/- {:.4f}".format(np.mean(test_mif1),np.std(test_mif1)))
        fh.write('\r\n')
        fh.flush()
        fh.close()


if __name__ == '__main__':

        experiment = Experiment()
        experiment.train(args)
