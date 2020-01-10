import re
import torch
import numpy as np
import os
import graph
from scipy.sparse import csr_matrix
import argparse

def get_sparse_matrix_and_rel2Idx(graph):
    rel2idx = dict()
    rels = []

    data = []
    row = []
    col = []

    N = len(graph.pred2Node)

    for pred,node in graph.pred2Node.items():
        idx = node.idx
        rel2idx[pred] = idx
        for oedge in node.oedges:
            nIdx = oedge.idx
            if (oedge.w>0):
                data.append(oedge.w)
                row.append(idx)
                col.append(nIdx)

    for idx in range(N):
        rels.append(graph.nodes[idx].id)

    A = csr_matrix((data, (row, col)), shape=(N, N))
    return A,rel2idx,rels

def read_graphs(gpath, f_post_fix="_sim.txt", featIdx = 0, isCCG=True, lower=True):

    args = argparse.ArgumentParser(description='graph args')
    args.CCG = isCCG
    graph.Graph.featIdx = args.featIdx = featIdx
    args.saveMemory = False
    args.threshold = -1.0

    engG_dir_addr = gpath + "/"
    files = os.listdir(engG_dir_addr)
    files = list(np.sort(files))

    num_f = 0
    types2E = dict()
    types2rel2idx = dict()
    types2rels = dict()

    for f in files:
        if num_f % 50 == 0:
            print ("num files: ", num_f)
        thisGpath = engG_dir_addr + f

        if f_post_fix not in f or os.stat(thisGpath).st_size == 0:
            continue

        num_f += 1

        gr = graph.Graph(gpath=thisGpath, args=args, lower=lower)
        gr.set_Ws()

        E,rel2Idx,rels = get_sparse_matrix_and_rel2Idx(gr)

        types_ss = gr.types
        types1 = types_ss[0]+"#"+types_ss[1]
        types2E[types1] = E
        types2rel2idx[types1] = rel2Idx
        types2rels[types1] = rels

        types2 = types_ss[1] + "#" + types_ss[0]
        types2E[types2] = E
        types2rel2idx[types2] = rel2Idx
        types2rels[types2] = rels

    return types2E, types2rel2idx, types2rels



#If we want to learn jointly, we should use randWalk code which gives E matrix, otherwise, we use graphs
#e2 is just given for reporting purposes
def getPredEnt(model, e1, e2, rels_idx, vocab):
    K = 300
    print ("new batch")
    types2E, types2rel2idx, types2rels, allRels2idx = model.types2E, model.types2rel2idx, model.types2rels, model.allRels2idx

    cuda0 = torch.device('cuda:0')

    pred = torch.zeros(rels_idx.shape[0],model.num_entities, device = cuda0)
    if not isinstance(rels_idx,list):
        rels_idx = rels_idx.cpu().numpy()

    for j,this_rel_idx in enumerate(rels_idx):

        rel = vocab['rel'].idx2token[this_rel_idx[0]]
        rel_orig = rel

        print ("rel: ", rel)
        reversed = "_reverse" in rel
        rel = rel.replace("_reverse","")
        ss = rel.split("#")

        types = ss[1] + "#" + ss[2]
        raw_type1 = ss[1].replace("_1","").replace("_2","")
        raw_type2 = ss[2].replace("_1", "").replace("_2", "")

        if types in types2E and rel in types2rel2idx[types]:
            print ("types: ", types)
            E = types2E[types]
            rel2idx = types2rel2idx[types]
            this_rels = types2rels[types]
            thisCol = E[:, rel2idx[rel]]
            neighs = thisCol.nonzero()[0]
            neighVals = thisCol.data

            numNeighs = neighs.shape[0]
            print ("num neighs: ", numNeighs)

            if numNeighs > K or rel2idx[rel] not in neighs:
                sortIdx = np.argsort(neighVals)
                new_neighs = list()
                new_neighVals = list()
                if numNeighs>0:
                    for i in range(numNeighs-1, np.max([numNeighs-K-1,0]), -1):
                        new_neighs.append(neighs[sortIdx[i]])
                        new_neighVals.append(neighVals[sortIdx[i]])
                if rel2idx[rel] not in new_neighs:
                    new_neighs.append(rel2idx[rel])
                    if E[rel2idx[rel],rel2idx[rel]]!=0:
                        new_neighVals.append(E[rel2idx[rel],rel2idx[rel]])
                    else:
                        new_neighVals.append(1.0)
                neighs = np.array(new_neighs)
                neighVals = np.array(new_neighVals,dtype=float)
                numNeighs = neighs.shape[0]
        else:
            print ("doesn't have: ", types, " or doesn't have pred")
            E = csr_matrix([[1.0]])
            rel2idx = {rel: 0}
            this_rels = [rel]
            thisCol = E[:, rel2idx[rel]]
            neighs = thisCol.nonzero()[0]
            numNeighs = neighs.shape[0]
            neighVals = thisCol.data

        neighVals /= np.sum(neighVals)

        q_rels_idx = torch.zeros(numNeighs, 1, dtype=torch.long, device=cuda0)
        e_idx = torch.ones(numNeighs, 1, dtype=torch.long, device=cuda0) * e1[j, 0]

        for i, q in enumerate(neighs):#The original one can't be thing_2 thing_1, but its neighbor might be!
            neigh_rel = this_rels[q]
            this_reversed = reversed

            ss_neigh = neigh_rel.split("#")
            if ss_neigh[1]!=ss[1]:
                this_reversed = not this_reversed
                if raw_type1==raw_type2 and ss_neigh[1].endswith("_2"):
                    neigh_rel = ss_neigh[0] + "#" + raw_type1+"_1#"+raw_type1+"_2"
            if this_reversed:
                neigh_rel += "_reverse"
            rel_neigh_idx = allRels2idx[neigh_rel]
            q_rels_idx[i, 0] = rel_neigh_idx
            print ("neigh: ", neigh_rel)
            print ("prob: ", neighVals[i])

        pred_q = model.forward(e_idx, q_rels_idx,numNeighs)

        self_score = None

        for i, prob in enumerate(neighVals):

            pred[j, :] += pred_q[i, :] * prob
            if neighs[i] == rel2idx[rel]:
                self_score = pred_q[i, :]

        e1_str = vocab['e1'].idx2token[e1[j, 0].item()]
        e2_str = vocab['e1'].idx2token[e2[j, 0].item()]

        print ("scores for ", rel_orig, " ", e1_str, e2_str)
        print ("self score: ", self_score[e2[j, 0].item()].item())
        print ("sum score: ", pred[j, e2[j, 0].item()].item())

        print ("self score shape: ", self_score.shape)
        print ("pred shape: ", pred.shape)

        pred[j, :] = torch.max(pred[j, :], self_score)

    pred = torch.clamp(pred,max=1.0)
    return pred

def get_AllEntities(vocab):
    allTokens = []
    tokensSet = set()
    for i in range(vocab['e1'].num_token):
        tok = vocab['e1'].idx2token[i]
        if not tok in tokensSet:
            tokensSet.add(tok)
            allTokens.append(tok)
    return allTokens

def get_AllRels(vocab):
    allTokens = []
    tokensSet = set()
    for i in range(vocab['rel'].num_token):
        tok = vocab['rel'].idx2token[i]
        if not tok in tokensSet:
            tokensSet.add(tok)
            allTokens.append(tok)

    return allTokens










