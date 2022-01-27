import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
import torch.nn.init as init
import numpy as np


class PROTOTYPE(nn.Module):
    def __init__(self, interact_matrix, social_matrix, user_num, item_num, embedding_size, L, n_cluster, device):
        super(PROTOTYPE, self).__init__()

        self.L = L
        self.user_num = user_num
        self.item_num = item_num
        
        self.user_Embedding = nn.Embedding(user_num, embedding_size)
        self.item_Embedding = nn.Embedding(item_num, embedding_size)

        interact_matrix = interact_matrix.coalesce()
        self.R = interact_matrix.to(device).float()

        top_indices = torch.stack([interact_matrix.indices()[0], interact_matrix.indices()[1] + user_num])
        bottom_indices = torch.stack([interact_matrix.indices()[1] + user_num, interact_matrix.indices()[0]])

        indices = torch.cat([top_indices, bottom_indices], dim=-1)

        values = torch.cat([interact_matrix.values(), interact_matrix.values()], dim=-1).float()

        self.A = torch.sparse_coo_tensor(indices, values, (user_num+item_num, user_num+item_num)).to(device)

        degree = torch.sparse.sum(self.A, dim=1).to_dense()
        degree = torch.pow(degree, -0.5)
        degree[torch.isinf(degree)] = 0

        self.D_inverse = torch.diag(degree, diagonal=0).to_sparse().to(device)

        self.A_hat = torch.sparse.mm(torch.sparse.mm(self.D_inverse, self.A), self.D_inverse)

        self.mlp_users = nn.ModuleList([nn.Sequential(nn.Linear(embedding_size, n_cluster), nn.ReLU(), nn.Linear(n_cluster, n_cluster, bias=False), nn.Softmax()) for i in range(L)])
        self.mlp_items = nn.ModuleList([nn.Sequential(nn.Linear(embedding_size, n_cluster), nn.ReLU(), nn.Linear(n_cluster, n_cluster, bias=False), nn.Softmax()) for i in range(L)])

        self.user_A = torch.sparse.mm(self.R, self.R.t()).to_dense()
        degree_u = torch.sum(self.user_A, dim=1)
        self.user_D = torch.diag(degree_u, diagonal=0).to(device)

        self.item_A = torch.sparse.mm(self.R.t(), self.R).to_dense()
        degree_i = torch.sum(self.item_A, dim=1)
        self.item_D = torch.diag(degree_i, diagonal=0).to(device)


    def gumbel_softmax(self, logits, temperature, eps=1e-20):
        U = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device)
        sample_gumbel = -torch.log(-torch.log(U + eps) + eps)
        y = logits + sample_gumbel
        y = functional.softmax(y / temperature, dim=-1)
        return y

    def cut_loss(self, S, A, D):
        up = torch.mm(torch.mm(S.t(), A), S)
        up = torch.trace(up)

        down = torch.mm(torch.mm(S.t(), D), S)
        down = torch.trace(down)
        return -(up/down)

    def orthogonality_loss_term(self, S):
        K = S.shape[1]
        right_term = torch.eye(K).to(S.device) / np.sqrt(K)
        SS = torch.mm(S.t(), S)
        left_term = SS / torch.linalg.matrix_norm(SS)

        return torch.linalg.matrix_norm(left_term - right_term)



    def forward(self):
        user_embeddings = self.user_Embedding.weight
        item_embeddings = self.item_Embedding.weight

        loss_cut = 0

        E_initial = torch.cat([user_embeddings, item_embeddings], dim=0)

        final_user_embedding = user_embeddings
        final_item_embedding = item_embeddings

        cur_term = E_initial
        for i in range(self.L):
            cur_term = torch.sparse.mm(self.A_hat, cur_term)

            cur_user_embedding = cur_term[:self.user_num]
            cur_item_embedding = cur_term[self.user_num:]

            H_user = self.mlp_users[i](cur_user_embedding)
            H_item = self.mlp_items[i](cur_item_embedding)

            loss_cut += self.cut_loss(H_user, self.user_A, self.user_D) 
            loss_cut += self.orthogonality_loss_term(H_user)
            loss_cut += self.cut_loss(H_item, self.item_A, self.item_D) 
            loss_cut += self.orthogonality_loss_term(H_item)


            HHU = functional.softmax(torch.mm(H_user, H_user.t()), dim=-1)
            HHI = functional.softmax(torch.mm(H_item, H_item.t()), dim=-1)

            cur_user_embedding = torch.mm(HHU, cur_user_embedding) + cur_user_embedding
            cur_item_embedding = torch.mm(HHI, cur_item_embedding) + cur_item_embedding

            final_user_embedding = final_user_embedding + cur_user_embedding
            final_item_embedding = final_item_embedding + cur_item_embedding

            cur_term = torch.cat([cur_user_embedding, cur_item_embedding], dim=0)

        final_user_embedding = final_user_embedding / (self.L + 1)
        final_item_embedding = final_item_embedding / (self.L + 1)

        return final_user_embedding, final_item_embedding, loss_cut















            










