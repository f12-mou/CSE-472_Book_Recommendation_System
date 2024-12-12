'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.norm_adj = norm_adj
        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # Xavier initialization
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_size))),
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})
            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})
        
        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        # Regularization
        regularizer = (torch.norm(users) ** 2 +
                       torch.norm(pos_items) ** 2 +
                       torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def compute_contrastive_loss(self, orig_emb, aug_emb, pos_emb, aug_pos_emb, tau=0.1):
        """
        Computes contrastive loss (InfoNCE).
        """
        # Normalize embeddings
        orig_emb = F.normalize(orig_emb, p=2, dim=1)
        aug_emb = F.normalize(aug_emb, p=2, dim=1)

        # Similarity between original and augmented views (positive pairs)
        pos_sim = torch.exp(torch.sum(orig_emb * aug_emb, dim=-1) / tau)

        # Similarity between original embeddings and all others (negative pairs)
        all_sim = torch.exp(torch.matmul(orig_emb, aug_emb.t()) / tau).sum(dim=1)

        # Contrastive loss (negative log likelihood of positive pairs)
        contrastive_loss = -torch.log(pos_sim / all_sim).mean()

        return contrastive_loss

    def augment_embeddings(self, users, pos_items):
      # Apply node dropout to create augmented graph
      aug_A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout, self.sparse_norm_adj._nnz())
      aug_ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

      all_embeddings = [aug_ego_embeddings]

      for k in range(len(self.layers)):
          # Propagate through graph layers with augmentation
          aug_side_embeddings = torch.sparse.mm(aug_A_hat, aug_ego_embeddings)

          # Sum-based transformation
          aug_sum_embeddings = torch.matmul(aug_side_embeddings, self.weight_dict['W_gc_%d' % k]) + \
                             self.weight_dict['b_gc_%d' % k]

          # Bi-linear transformation
          aug_bi_embeddings = torch.mul(aug_ego_embeddings, aug_side_embeddings)
          aug_bi_embeddings = torch.matmul(aug_bi_embeddings, self.weight_dict['W_bi_%d' % k]) + \
                             self.weight_dict['b_bi_%d' % k]

          # Combine, apply non-linearity, and dropout
          aug_ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(aug_sum_embeddings + aug_bi_embeddings)
          aug_ego_embeddings = nn.Dropout(self.mess_dropout[k])(aug_ego_embeddings)

          all_embeddings += [aug_ego_embeddings]

      # Concatenate all layers' embeddings
      aug_all_embeddings = torch.cat(all_embeddings, 1)

      # Normalize embeddings
      aug_all_embeddings = F.normalize(aug_all_embeddings, p=2, dim=1)

      # Extract user and item embeddings
      aug_u_g_embeddings = aug_all_embeddings[:self.n_user, :][users, :]
      aug_i_g_embeddings = aug_all_embeddings[self.n_user:, :][pos_items, :]

      return aug_u_g_embeddings, aug_i_g_embeddings


    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # Transformed sum messages of neighbors
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + \
                             self.weight_dict['b_gc_%d' % k]

            # Bi-linear messages of neighbors
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + \
                            self.weight_dict['b_bi_%d' % k]

            # Non-linear activation and dropout
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # Normalize embeddings
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :][users, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
