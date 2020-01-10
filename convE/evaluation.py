# Modified version of the code at https://github.com/TimDettmers/ConvE

import torch
import numpy as np
import datetime

from spodernet.utils.global_config import Config
from spodernet.utils.logger import Logger
from collections import defaultdict
from .util import *

log = Logger('evaluation{0}.py.txt'.format(datetime.datetime.now()))

def ranking_and_hits(model, dev_rank_batcher, vocab, name, all_triples_path = None, N_freq = 20):
    log.info('')
    log.info('-'*50)
    log.info(name)
    log.info('-'*50)
    log.info('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []

    hits_left_inf = []
    hits_right_inf = []
    hits_inf = []
    ranks_inf = []
    ranks_left_inf = []
    ranks_right_inf = []

    if all_triples_path:
        freq_ents = get_freq_entities(all_triples_path, N_freq)
    else:
        freq_ents = None

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

        hits_left_inf.append([])
        hits_right_inf.append([])
        hits_inf.append([])

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']
        rel_reverse = str2var['rel_eval']
        e2_multi1 = str2var['e2_multi1'].float()
        e2_multi2 = str2var['e2_multi2'].float()

        pred1 = model.forward(e1, rel)
        pred2 = model.forward(e2, rel_reverse)

        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data

        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(Config.batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            num = e1[i, 0].item()
            # save the prediction that is relevant
            target_value1 = pred1[i,e2[i, 0].item()].item()
            target_value2 = pred2[i,e1[i, 0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        for i in range(Config.batch_size):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i]==e2[i, 0].item())[0][0]
            rank2 = np.where(argsort2[i]==e1[i, 0].item())[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)

            is_infreq = freq_ents and (vocab['e1'].idx2token[e1[i,0].item()] not in freq_ents) and (vocab['e1'].idx2token[e2[i, 0].item()] not in freq_ents)

            if freq_ents and is_infreq:

                ranks_inf.append(rank1 + 1)
                ranks_left_inf.append(rank1 + 1)
                ranks_inf.append(rank2 + 1)
                ranks_right_inf.append(rank2 + 1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

            if freq_ents and is_infreq:
                for hits_level in range(10):

                    if rank1 <= hits_level:
                        hits_inf[hits_level].append(1.0)
                        hits_left_inf[hits_level].append(1.0)
                    else:
                        hits_inf[hits_level].append(0.0)
                        hits_left_inf[hits_level].append(0.0)
                    if rank2 <= hits_level:
                        hits_inf[hits_level].append(1.0)
                        hits_right_inf[hits_level].append(1.0)
                    else:
                        hits_inf[hits_level].append(0.0)
                        hits_right_inf[hits_level].append(0.0)

        dev_rank_batcher.state.loss = [0]

    for i in range(10):
        log.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
        log.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
        log.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
    log.info('Mean rank left: {0}', np.mean(ranks_left))
    log.info('Mean rank right: {0}', np.mean(ranks_right))
    log.info('Mean rank: {0}', np.mean(ranks))
    log.info('Mean reciprocal rank left: {0}', np.mean(1./np.array(ranks_left)))
    log.info('Mean reciprocal rank right: {0}', np.mean(1./np.array(ranks_right)))
    log.info('Mean reciprocal rank: {0}', np.mean(1./np.array(ranks)))

    if freq_ents:
        for i in range(10):
            log.info('Hits left infreq @{0}: {1}'.format(i + 1, np.mean(hits_left_inf[i])))
            log.info('Hits right infreq @{0}: {1}'.format(i + 1, np.mean(hits_right_inf[i])))
            log.info('Hits infreq @{0}: {1}'.format(i + 1, np.mean(hits_inf[i])))
        log.info('Mean rank left infreq: {0}', np.mean(ranks_left_inf))
        log.info('Mean rank right infreq: {0}', np.mean(ranks_right_inf))
        log.info('Mean rank infreq: {0}', np.mean(ranks_inf))
        log.info('Mean reciprocal rank left infreq: {0}', np.mean(1. / np.array(ranks_left_inf)))
        log.info('Mean reciprocal rank right infreq: {0}', np.mean(1. / np.array(ranks_right_inf)))
        log.info('Mean reciprocal rank infreq: {0}', np.mean(1. / np.array(ranks_inf)))


#E is the ent graph matrix in csr format. E is already normalized and only the first view are selected, etc
def ranking_and_hits_entGraph(model, dev_rank_batcher, vocab, name, all_triples_path = None, N_freq = 20):
    log.info('')
    log.info('-'*50)
    log.info(name)
    log.info('-'*50)
    log.info('')
    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []

    hits_left_inf = []
    hits_right_inf = []
    hits_inf = []
    ranks_inf = []
    ranks_left_inf = []
    ranks_right_inf = []

    if all_triples_path:
        freq_ents = get_freq_entities(all_triples_path, N_freq)
    else:
        freq_ents = None

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

        hits_left_inf.append([])
        hits_right_inf.append([])
        hits_inf.append([])

    for i, str2var in enumerate(dev_rank_batcher):
        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']

        rel_reverse = str2var['rel_eval']
        e2_multi1 = str2var['e2_multi1'].float()
        e2_multi2 = str2var['e2_multi2'].float()

        pred1 = getPredEnt(model, e1, e2, rel , vocab)
        pred2 = getPredEnt(model, e2, e1, rel_reverse, vocab)


        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data

        e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data
        for i in range(Config.batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[i].long()
            filter2 = e2_multi2[i].long()

            # save the prediction that is relevant
            target_value1 = pred1[i,e2[i, 0].item()].item()
            target_value2 = pred2[i,e1[i, 0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[i][filter1] = 0.0
            pred2[i][filter2] = 0.0
            # write base the saved values
            pred1[i][e2[i]] = target_value1
            pred2[i][e1[i]] = target_value2


        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        argsort1 = argsort1.cpu().numpy()
        argsort2 = argsort2.cpu().numpy()
        for i in range(Config.batch_size):
            # find the rank of the target entities
            rank1 = np.where(argsort1[i]==e2[i, 0].item())[0][0]
            rank2 = np.where(argsort2[i]==e1[i, 0].item())[0][0]
            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)

            is_infreq = freq_ents and (vocab['e1'].idx2token[e1[i, 0].item()] not in freq_ents) and (vocab['e1'].idx2token[e2[i, 0].item()] not in freq_ents)

            if freq_ents and is_infreq:
                ranks_inf.append(rank1 + 1)
                ranks_left_inf.append(rank1 + 1)
                ranks_inf.append(rank2 + 1)
                ranks_right_inf.append(rank2 + 1)

            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

            if freq_ents and is_infreq:
                for hits_level in range(10):

                    if rank1 <= hits_level:
                        hits_inf[hits_level].append(1.0)
                        hits_left_inf[hits_level].append(1.0)
                    else:
                        hits_inf[hits_level].append(0.0)
                        hits_left_inf[hits_level].append(0.0)
                    if rank2 <= hits_level:
                        hits_inf[hits_level].append(1.0)
                        hits_right_inf[hits_level].append(1.0)
                    else:
                        hits_inf[hits_level].append(0.0)
                        hits_right_inf[hits_level].append(0.0)

        dev_rank_batcher.state.loss = [0]

    for i in range(10):
        log.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
        log.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
        log.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
    log.info('Mean rank left: {0}', np.mean(ranks_left))
    log.info('Mean rank right: {0}', np.mean(ranks_right))
    log.info('Mean rank: {0}', np.mean(ranks))
    log.info('Mean reciprocal rank left: {0}', np.mean(1./np.array(ranks_left)))
    log.info('Mean reciprocal rank right: {0}', np.mean(1./np.array(ranks_right)))
    log.info('Mean reciprocal rank: {0}', np.mean(1./np.array(ranks)))


    if freq_ents:
        for i in range(10):
            log.info('Hits left infreq @{0}: {1}'.format(i + 1, np.mean(hits_left_inf[i])))
            log.info('Hits right infreq @{0}: {1}'.format(i + 1, np.mean(hits_right_inf[i])))
            log.info('Hits infreq @{0}: {1}'.format(i + 1, np.mean(hits_inf[i])))
        log.info('Mean rank left infreq: {0}', np.mean(ranks_left_inf))
        log.info('Mean rank right infreq: {0}', np.mean(ranks_right_inf))
        log.info('Mean rank infreq: {0}', np.mean(ranks_inf))
        log.info('Mean reciprocal rank left infreq: {0}', np.mean(1. / np.array(ranks_left_inf)))
        log.info('Mean reciprocal rank right infreq: {0}', np.mean(1. / np.array(ranks_right_inf)))
        log.info('Mean reciprocal rank infreq: {0}', np.mean(1. / np.array(ranks_inf)))


def get_freq_entities(triples_path, N=20):
    f = open(triples_path)
    e2count = defaultdict(int)
    for line in f:
        # print (line)
        try:
            line = line.strip()
            ss = line.split("\t")
            if len(ss)!=5 and len(ss)!=3:
                # print ("bad line: ", line)
                continue
            e2count[ss[0]] += 1
            e2count[ss[2]] += 1
        except Exception as e:
            # print ("bad line: ", line)
            # print (e)
            continue
    pairs = []
    for e,c in e2count.items():
        pairs.append([e,c])

    pairs_sorted = sorted(pairs,key= lambda pair: pair[1],reverse=True)
    freq_ents = []
    # print ("freqs: ")
    for i in range(N):
        # print (pairs_sorted[i])
        freq_ents.append(pairs_sorted[i][0])
    return freq_ents


def get_all_ent_pairs(triples_path):
    f = open(triples_path)
    e_pairs = defaultdict(set)
    e_pairs_rev = defaultdict(set)
    for line in f:
        try:
            line = line.strip()
            ss = line.split("\t")
            if len(ss)!=5:
                # print ("bad line: ", line)
                continue
            e_pairs[ss[0]].add(ss[2])
            e_pairs_rev[ss[2]].add(ss[0])

        except:
            continue
    return e_pairs, e_pairs_rev


def compute_probs(model, dev_rank_batcher, vocab, name, fout,triples_path):#the usual one
    log.info('')
    log.info('-'*50)
    log.info(name)
    log.info('-'*50)
    log.info('')

    e_pairs, e_pairs_rev = get_all_ent_pairs(triples_path)
    ent2idx = {}
    for i in vocab['e1'].idx2token:
        ent2idx[vocab['e1'].idx2token[i]] = i

    seen_rel_e_pairs = set()

    for i, str2var in enumerate(dev_rank_batcher):

        e1 = str2var['e1']
        e2 = str2var['e2']
        rel = str2var['rel']
        rel_reverse = str2var['rel_eval']


        pred1 = model.forward(e1, rel)
        pred2 = model.forward(e2, rel_reverse)

        pred1, pred2 = pred1.data, pred2.data

        e1, e2 = e1.data, e2.data

        for i in range(Config.batch_size):#rel1[i],e1[i] in the batch

            rel_name = vocab['rel'].idx2token[rel[i].item()]
            rel_name_reverse = vocab['rel'].idx2token[rel_reverse[i].item()]
            e1_name = vocab['e1'].idx2token[e1[i,0].item()]

            rel_e1_pair = rel_name + "#" + e1_name
            this_values1 = pred1[i,:].detach().cpu().numpy()
            this_values2 = pred2[i, :].detach().cpu().numpy()

            if rel_e1_pair not in seen_rel_e_pairs:
                fout.write(rel_name + " " + e1_name + " ")

                # If you only want the seen ent-pairs that you're lucky to have the first entity with the relation!
                this_e_ps = []
                for e2_name in e_pairs[e1_name]:
                    j = ent2idx[e2_name]
                    prob = this_values1[j]
                    this_e_ps.append([e2_name,prob])

                this_e_ps = sorted(this_e_ps,key=lambda x:x[1],reverse=True)
                for e2_name,prob in this_e_ps:
                    fout.write(e2_name +" " + str(prob)+" ")

                fout.write("\n")
            seen_rel_e_pairs.add(rel_e1_pair)

            e2_name = vocab['e1'].idx2token[e2[i, 0].item()]
            rel_reverse_e2_pair = rel_name_reverse + "#" + e2_name

            if rel_reverse_e2_pair not in seen_rel_e_pairs:
                fout.write(rel_name_reverse + " " + e2_name + " ")
                this_e_ps = []
                for e1_name in e_pairs_rev[e2_name]:
                    j = ent2idx[e1_name]
                    prob = this_values2[j]
                    this_e_ps.append([e1_name, prob])


                this_e_ps = sorted(this_e_ps, key=lambda x: x[1], reverse=True)
                for e1_name, prob in this_e_ps:
                    fout.write(e1_name +" " + str(prob)+" ")

                fout.write("\n")

            seen_rel_e_pairs.add(rel_reverse_e2_pair)

        dev_rank_batcher.state.loss = [0]
