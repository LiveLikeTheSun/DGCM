from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
# from pygcn.utils import load_data, accuracy
import math
from torch.utils.data import TensorDataset, DataLoader
# from model1 import *
from DGCM import *
from sklearn.preprocessing import MinMaxScaler
import random
import os
from torch.utils.tensorboard import SummaryWriter
# from gpu_mem_track import MemTracker
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from ipdb import set_trace

def compute_perplexity(pred_logits, TRUE_CLICKS):
    '''Compute the perplexity'''
    perplexity_at_rank = [0.0] * 10
    for rank in range(len(TRUE_CLICKS)):
        if TRUE_CLICKS[rank] == 1:
            perplexity_at_rank[rank] = torch.log2(pred_logits[rank] + 1e-30)
        else:
            perplexity_at_rank[rank] = torch.log2(1. - pred_logits[rank] + 1e-30)
    return torch.tensor(perplexity_at_rank)


def compute_ll(pred_logits, TRUE_CLICKS):
    # compute loglikelihood
    perplexity_at_rank = [0.0] * 10
    for rank in range(len(TRUE_CLICKS)):
        if TRUE_CLICKS[rank] == 1:
            perplexity_at_rank[rank] = torch.log(pred_logits[rank] + 1e-30)
        else:
            perplexity_at_rank[rank] = torch.log(1. - pred_logits[rank] + 1e-30)
    return torch.tensor(perplexity_at_rank)


def dcg_(ranking_relevances):
    # Compute the DCG for a given ranking_relevances
    return sum([(2 ** relevance - 1) / math.log(rank + 2, 2) for rank, relevance in enumerate(ranking_relevances)])


def evaluation(clicks, probablity):
    perplexity_at_rank = torch.tensor([0.0] * 10)  # 6245930
    tmp_ll = torch.tensor([0.0] * 10)
    for i in range(len(clicks)):
        probs = []
        for x in range(i, len(clicks) * 10, len(clicks)):
            if len(probs) < 10:
                probs.append(probablity[x])
        one_perplexity = compute_perplexity(probs, clicks[i])
        perplexity_at_rank = perplexity_at_rank + one_perplexity
        tmp = compute_ll(probs, clicks[i])
        tmp_ll = tmp_ll + tmp
    loglikelihood = tmp_ll.sum() / (len(clicks) * 10)
    perplexity = torch.tensor(0.0)
    for i in range(len(perplexity_at_rank)):
        perplexity += 2 ** ((- perplexity_at_rank[i]) / len(clicks))
    perplexity = perplexity / 10

    print('loglikelihood:', loglikelihood)
    print('perplexity:', perplexity)
    return loglikelihood, perplexity

def rank(clicks, probablity, true_relevance):
    test_relevance = []
    for i in range(len(clicks)):
        probs = []
        for x in range(i, len(clicks) * 10, len(clicks)):
            if len(probs) < 10:
                probs.append(probablity[x].detach().cpu().numpy())
        test_relevance.append(probs)
    #np.save('model/gat/add_click/test_pred_rele', test_relevance)
    trunc_levels = [1, 3, 5, 10]
    ndcgs, cnt_useless_session, cnt_usefull_session = {}, {}, {}
    for k in trunc_levels:
        ndcgs[k] = 0.0
        cnt_useless_session[k] = 0
        cnt_usefull_session[k] = 0

    for relevances, true_relevances in zip(test_relevance, true_relevance):
        pred_rels = {}
        for idx, relevance in enumerate(relevances):
            pred_rels[idx] = relevance

        for k in trunc_levels:
            ideal_ranking_relevance = sorted(true_relevances, reverse=True)[:k]
            ranking = sorted([idx for idx in pred_rels], key=lambda idx: pred_rels[idx], reverse=True)
            ranking_relevances = [true_relevances[idx] for idx in ranking[:k]]
            dcg = dcg_(ranking_relevances)
            idcg = dcg_(ideal_ranking_relevance)
            ndcg = dcg / idcg if idcg > 0 else 1.0
            if idcg == 0:
                cnt_useless_session[k] += 1
            else:
                ndcgs[k] += ndcg
                cnt_usefull_session[k] += 1
    for k in trunc_levels:
        ndcgs[k] /= cnt_usefull_session[k]
    print(ndcgs)
    return ndcgs

def adjust_learning_rate(optimizer, decay_rate=0.5):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train(epoch, data):
    model.train()
    criterion = torch.nn.BCELoss()
    train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    global learning_rate
    global patience

    all_loglikelihood, all_perplexity = [], []
    # test_loss, loglikelihood, perplexity = test()
    # if round(loglikelihood.item(), 4) not in all_loglikelihood:
    #     all_loglikelihood.append(round(loglikelihood.item(), 4))
    # if round(perplexity.item(), 4) not in all_perplexity:
    #     all_perplexity.append(round(perplexity.item(), 4))
    for step, batch in enumerate(train_loader):
        t = time.time()
        sessions = []
        train_fea = []
        train_adj = [[], []]
        doc_id = []
        title_id = []
        img_id = []
        sessions = batch['session']
        querys = batch['query'].tolist()
        # print(batch)
        num = 0
        for se in sessions: # 得到一个batch的feature和邻接矩阵
            tmp_title_id = []
            sess_feas = np.zeros((len(all_sess_img_id[se])+31, 216)).tolist()
            show_idx = all_sess_show_idx[se]
            for i in show_idx:
                sess_feas[21+show_idx.index(i)][:160] = query_sni2fea[querys[num]][i]
                # sess_feas[11+show_idx.index(i)][:160] = query_title2fea[querys[num]][i]
            for i in all_sess_img_id[se]:
                sess_feas[31+all_sess_img_id[se].index(i)][:160] = img_fea[i]
            train_fea += sess_feas
            for i in show_idx:
                tmp_title_id.append(query_title2id[querys[num]][i])
            title_id.append(tmp_title_id)
            
            num += 1
            if len(train_adj[0]) > 0:
                length = max(train_adj[0]) + 1
                # length = max([max(train_adj[0]), max(train_adj[1])])+1
            else:
                length = 0

            for node_adj_source in all_sess_adj[se][0]:
                train_adj[0].append(node_adj_source+length)
            
            for node_adj_target in all_sess_adj[se][1]:
                train_adj[1].append(node_adj_target+length)

            doc_id.append([_+length for _ in range(1, 11)])

        loss = torch.tensor(0.0, requires_grad=True).cuda()
        x = torch.tensor(np.array(train_fea)).to(torch.float32).cuda()  # torch.tensor(np.array(x)).to(torch.float32).cuda()
        urls_id = []
        for mm in range(args.batch_size):
            urls = []
            for nn in batch['url_ids']:
                urls.append(nn.tolist()[mm])
            urls_id.append(urls)
        
        for i in range(10):
            result_id = []  # 每个query下第i个结果的id
            for docs in doc_id:
                result_id.append(docs[i])
            result_clicks = batch['click'][i]  # 每个query下第i个结果的id对应的click
            result_click = [[_] for _ in batch['click'][i + 1].tolist()]
              
            hidden_state, output, x_part = model(i, x, torch.tensor(train_adj).cuda(), torch.tensor(doc_id).cuda(), result_clicks.cuda(), batch['query'].cuda(), torch.tensor(urls_id).cuda(), torch.tensor(title_id).cuda())  # 运行模型，输入参数（features,adj）即特征和邻接矩阵
            loss += criterion(output[result_id], torch.tensor(result_click).float().cuda()) # index index_click
            
            x = torch.cat((x_part, hidden_state), dim=1)
        # print('------------------------------------------------')
        optimizer.zero_grad()  # 把模型中参数的梯度设为0
        loss = loss / 10
        loss.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'batch: {:04d}'.format(step),
              'loss_train: {:.4f}'.format(loss),
              'time: {:.4f}s'.format(time.time() - t))

        writer.add_scalar(tag="loss/train", scalar_value=loss,
                          global_step=epoch * 14475 + step)
        
        if step > 0 and step % 300 == 0:
            tmp_test_loss, tmp_loglikelihood, tmp_perplexity = test()
            if tmp_loglikelihood > max(all_loglikelihood) or tmp_perplexity < min(all_perplexity):
                patience = 0
                torch.save(model, 'output/DGCM')
                if round(tmp_loglikelihood.item(), 4) not in all_loglikelihood:
                    all_loglikelihood.append(round(tmp_loglikelihood.item(), 4))
                if round(tmp_perplexity.item(), 4) not in all_perplexity:
                    all_perplexity.append(round(tmp_perplexity.item(), 4))
            
            else:
                patience += 1
            if patience >= args.patience:
                adjust_learning_rate(optimizer, args.lr_decay)
                learning_rate *= args.lr_decay
                writer.add_scalar(tag='loss/lr', scalar_value=learning_rate, global_step=epoch * 14475 + step)
                loglikelihood = tmp_loglikelihood
                perplexity = tmp_perplexity
                patience = 0
                args.patience += 1
            writer.add_scalar(tag="loss/test", scalar_value=tmp_test_loss,
                          global_step=epoch * 14475 + step)

# 定义测试的一个函数
def test(idx_test):
    # model = torch.load('model/202302/44_emb_id_title')
    model.eval()
    criterion = torch.nn.BCELoss().cuda()
    results_id = []
    clicks = []
    true_clicks = []
    querys = []
    sessions = []
    url_idx = []
    idx = []
    for j in idx_test:
        sessions.append(j['session'])
        querys.append(j['query'])
        results_id.append(j['result_id'])
        clicks.append(j['click'])
        true_clicks.append(j['click'][1:])
        url_idx.append(j['url_ids'])
    print(len(sessions))
    test_fea = []
    test_adj = [[], []]
    doc_id = []
    num = 0
    title_id = []
    for se in sessions: # 得到一个batch的feature和邻接矩阵
        tmp_title_id = []
        sess_feas = np.zeros((len(all_sess_img_id[se])+31, 216)).tolist()
        show_idx = all_sess_show_idx[se]
        for i in show_idx:
            sess_feas[21+show_idx.index(i)][:160] = query_sni2fea[querys[num]][i]
            # sess_feas[11+show_idx.index(i)][:160] = query_title2fea[querys[num]][i]
        for i in all_sess_img_id[se]:
            sess_feas[31+all_sess_img_id[se].index(i)][:160] = img_fea[i]
        test_fea += sess_feas
        for i in show_idx:
            tmp_title_id.append(query_title2id[querys[num]][i])
        title_id.append(tmp_title_id)

        num += 1
        if len(test_adj[0]) > 0:
            length = max(test_adj[0]) + 1
            # length = max([max(test_adj[0]), max(test_adj[1])])+1
        else:
            length = 0

        for node_adj_source in all_sess_adj[se][0]:
            test_adj[0].append(node_adj_source+length)
        for node_adj_target in all_sess_adj[se][1]:
            test_adj[1].append(node_adj_target+length)

        doc_id.append([_+length for _ in range(1, 11)])
        
    x = torch.tensor(np.array(test_fea)).to(torch.float32).cuda()
    probablity = []
    hidden = []
    loss = torch.tensor(0.0).cuda()
    for i in range(10):
        result_click = []  # get i-th docs and its clicks
        result_clicks = []
        result_id = []
        for cli in clicks:
            result_clicks.append(cli[i])
        for cli in true_clicks:
            result_click.append([cli[i]])
        
        for docs in doc_id:
            result_id.append(docs[i])

        
        hidden_state, output, x_part = model(i, x, torch.tensor(test_adj).cuda(), torch.tensor(doc_id).cuda(), torch.tensor(result_clicks).cuda(), torch.tensor(querys).cuda(), torch.tensor(url_idx).cuda(), torch.tensor(title_id).cuda())  # , False
        # hidden_state = hidden_state.detach().cpu()
        # hidden += hidden_state[result_id].tolist()
        output = output.detach().cpu()
        probablity += output[result_id]
        loss += criterion(output[result_id].cuda(), torch.tensor(result_click).float().cuda())
        # print('test_loss:', loss.item())
        x = torch.cat((x_part, hidden_state), dim=1)

    print('test_loss:', loss.item() / 10)
    #np.save('model/gat/add_click/true_rele_allq', true_relevance)
    loglikelihood, perplexity = evaluation(true_clicks, probablity)
    # rank(clicks, probablity, true_relevance)
    return loss.item() / 10, loglikelihood, perplexity

# t_total = time.time()
# writer = SummaryWriter(log_dir="output/writer/")
# writer.add_scalar(tag='loss/lr', scalar_value=learning_rate, global_step=0)
# # 定义了训练多少回合
# for epoch in range(args.epochs): # args.epochs
#     # test()
#     train(epoch)
# writer.close()
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

