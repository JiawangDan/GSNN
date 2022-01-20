import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import math
import time

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h) + x2


class MergeLayer_output(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3= 1024, dim4=1, drop_out=0.2):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 * 4, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim3)
    self.fc3 = torch.nn.Linear(dim3, dim2)
    self.fc4 = torch.nn.Linear(dim2 , dim4 )
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop_out)

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2, x3, x4):
    x = torch.cat([x1, x2, x3, x4], dim=-1)
    h = self.act(self.fc1(x))
    h = self.act(self.fc2(h))
    h = self.dropout(self.act(self.fc3(h)))
    h = self.fc4(h)
    return h



class MergeLayer_concat1(torch.nn.Module):
  def __init__(self, dim1, dim2):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1*9, dim2)
    self.fc2 = torch.nn.Linear(dim2, dim2)

    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    h = self.act(self.fc1(x))
    h = self.act(self.fc2(h))
    return h



class Node_feat_process(torch.nn.Module):
  def __init__(self, dim1, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1):
    x = x1
    h = self.act(self.fc1(x))
    return self.fc2(h)

class Edge_feat_process(torch.nn.Module):
  def __init__(self, dim1, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    h = self.act(self.fc1(x))
    return self.fc2(h)


class Coattention_process(torch.nn.Module):
  def __init__(self, dims):
    super().__init__()
    self.dims = dims
    self.heads = 8
    self.dim_head = dims // self.heads
    self.scale = self.dim_head ** (-0.5)
    self.to_q = torch.nn.Linear(self.dims, self.dims, bias=True)
    self.to_kv = torch.nn.Linear(self.dims, self.dims * 2, bias=True)
    self.to_out = torch.nn.Linear(self.dims, self.dims, bias=True)

    torch.nn.init.xavier_normal_(self.to_q.weight)
    torch.nn.init.xavier_normal_(self.to_kv.weight)
    torch.nn.init.xavier_normal_(self.to_out.weight)

  def forward(self, src, seq):
    q = self.to_q(src)
    kv_out = self.to_kv(seq)
    k, v = torch.chunk(kv_out, 2, dim=-1)

    q = self.merge_heads(torch.unsqueeze(q, dim=1), self.dims)
    k = self.merge_heads(k, self.dims)
    v = self.merge_heads(v, self.dims)

    dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
    attn = torch.softmax(dots, dim=-1)

    out = torch.einsum('bhij,bhjd->bhid', attn, v)
    out = out.permute([0, 2, 1, 3])
    out = torch.reshape(out, [-1, self.dims])
    logit = self.to_out(out)
    return logit

  def merge_heads(self, x, dim):
    x = torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2]//dim, dim])
    x = x.permute([0, 2, 1, 3])
    return x