'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time


if __name__ == '__main__':

    if args.gpu_id == -1:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    
    print("Model created")

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        print("Epoch started")
        t1 = time()
        loss, mf_loss, emb_loss, contrastive_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            #print("Inside the innner for loop")
            
            # Original sampled data
            users, pos_items, neg_items = data_generator.sample()
            
            # Original embeddings
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users, 
                                                                           pos_items, 
                                                                           neg_items, 
                                                                           drop_flag=args.node_dropout_flag)
            #print("Found original embeddings")
            #print(f"u_g_embeddings shape: {u_g_embeddings.shape}")
            #print(f"pos_i_g_embeddings shape: {pos_i_g_embeddings.shape}")
            #print(f"neg_i_g_embeddings shape: {neg_i_g_embeddings.shape}")
            
            # Augmented embeddings
            aug_u_g_embeddings, aug_pos_i_g_embeddings = model.augment_embeddings(users, pos_items)
            #print("Found augmented embeddings")
            #print(f"aug_u_g_embeddings shape: {aug_u_g_embeddings.shape}")
            #print(f"aug_pos_i_g_embeddings shape: {aug_pos_i_g_embeddings.shape}")
            
            # Compute BPR loss
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            #print("BPR loss computed")

            # Compute Contrastive Loss
            batch_contrastive_loss = model.compute_contrastive_loss(u_g_embeddings, 
                                                                    aug_u_g_embeddings, 
                                                                    pos_i_g_embeddings, 
                                                                    aug_pos_i_g_embeddings)
            #print("Contrastive loss computed")

            # Total Loss
            total_batch_loss = batch_loss + args.contrastive_weight * batch_contrastive_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            contrastive_loss += batch_contrastive_loss

            #print("Inner for loop ended with at least one batch")

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
              total_loss = loss + args.contrastive_weight * contrastive_loss
              perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f], total_loss=[%.5f]' % (
              epoch, time() - t1, total_loss, mf_loss, emb_loss, args.contrastive_weight * contrastive_loss, total_loss)
              print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, contrastive_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')
        

        print("Epoch ended")

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
