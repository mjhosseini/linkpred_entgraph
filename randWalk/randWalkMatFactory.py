import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
import os
from constants.constants import ConstantsRWalk
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from collections import OrderedDict, defaultdict
from randWalk.randWalkMat import RandWalkMatrix
import argparse

class RandWalkMatrixFactory:

    uniqueRandWalkMatrixFactory = None

    def __init__(self):

        RandWalkMatrixFactory.uniqueRandWalkMatrixFactory = self
        self.numAllSeenEdges = 0
        self.numAllStoredTriples = 0
        self.typeToOrderedType = {}
        self.typesToGraphs = OrderedDict()
        self.entToType = {}
        if ConstantsRWalk.embsPath:
            self.relsToEmbed, _, _ = self.loadEmbeddings(ConstantsRWalk.embsPath)

        self.allTriples = self.loadAllTriples(ConstantsRWalk.allTriplesPath)
        self.tripleToScore = self.loadTriplesToScore(ConstantsRWalk.triples2scoresPath)


    def init_matrix(self):
        for types,rwMat in self.typesToGraphs.items():
            rwMat.init_matrix()

    #both init matrix and does the walk
    def performWalks(self):
        if not os.path.isdir(ConstantsRWalk.simsFolder):
            os.mkdir(ConstantsRWalk.simsFolder)
        for types,rwMat in self.typesToGraphs.items():
            rwMat.init_matrix()
            rwMat.setRandWalkMat(ConstantsRWalk.L)
            rwMat.writeResults()
            print ("random walk done for:", types)


    def get_orderedType(self, type1, type2, insert=True):
        types1 = type1+"#"+type2
        types2 = type2+"#"+type1

        if types1 in self.typeToOrderedType:
            return self.typeToOrderedType[types1]
        elif not insert:
            return None
        else:
            self.typeToOrderedType[types1] = types1
            self.typeToOrderedType[types2] = types1
            return types1

    def loadEmbeddings(self,fname):
        x2emb = {}
        f = open(fname)
        ent2idx = dict()

        for line in f:
            try:
                line = line.strip()
                ss = line.split("\t")
                x = ss[0]
                embVec = ss[1][1:-1]
                a = np.array([np.float(d) for d in embVec.split()])
                embs = np.ndarray(shape=(1,a.shape[0]))
                embs[0,:] = a
                ent2idx[x] = len(x2emb)
                x2emb[x] = embs

            except:
                traceback.print_exc()
                pass

        emb_size = a.shape[0]
        print (str(len(x2emb)) + " embeddings loaded")

        return x2emb, ent2idx, emb_size

    def loadAllTriples(self, fname):
        f = open(fname)
        ret = defaultdict(int)

        idx = 0

        for line in f:
            # if idx==100000:#used for debugging
            #     break
            idx += 1
            line = line.strip()
            ss = line.split("\t")
            if ConstantsRWalk.ptyped and ((not ConstantsRWalk.unary and len(ss[1].split("#"))!=3) or (ConstantsRWalk.unary and len(ss[0].split("#"))!=2)):
                continue
            try:
                if ConstantsRWalk.useFreq:
                    if not ConstantsRWalk.unary:
                        ret[ss[0] + "#" + ss[1] + "#" + ss[2]] = np.int(ss[5])
                    else:
                        ret[ss[0] + "#" + ss[1]] = np.int(ss[3])
                else:
                    if not ConstantsRWalk.unary:
                        ret[ss[0] + "#" + ss[1] + "#" + ss[2]] = 1
                    else:
                        ret[ss[0] + "#" + ss[1]] = 1
                if ConstantsRWalk.entType:
                    if not ConstantsRWalk.unary:
                        self.entToType[ss[0]] = ss[3]
                        self.entToType[ss[2]] = ss[4]
                    else:
                        self.entToType[ss[1].split("#")[0]] = ss[2]
            except Exception as e:
                print (e)
                continue
        print (str(len(ret)) + " triples loaded")
        return ret

    def loadTriplesToScore(self,fname):

        ret = defaultdict(np.double)

        if ConstantsRWalk.unary:
            idx = 0
            for predArg,count in self.allTriples.items():
                # if (idx==100000):#used for debugging!
                #     break
                ret[predArg] = 1
                ss = predArg.split("#")
                pred = ss[0]+"#"+ss[1]
                if not ConstantsRWalk.timeStamp:
                    arg = ss[2]
                else:
                    arg = ss[2]+"#"+ss[3]
                type = ss[1]
                self.add_triple_to_graph(pred, arg, 1, type, count)
                idx += 1
            return ret

        f = open(fname)

        index = 0

        for line in f:
            line = line.strip()
            try:
                index += 1

                # if index >= 100000:#used for debugging!
                #     break

                if index % 100000 == 0:
                    print (index)


                ss = line.strip().split(" ")
                pred = ss[0]
                reverse = pred.endswith("reverse")
                pred = pred.replace("_reverse", "")
                e1 = ss[1]

                N_ap = 0
                #num arg- pairs( if existing) read other than th original ones

                if ConstantsRWalk.ptyped:
                    pred_ss = pred.split("#")
                    rawPred = pred_ss[0]
                    type1_raw = pred_ss[1].replace("_1","").replace("_2","")
                    type2_raw = pred_ss[2].replace("_1","").replace("_2","")
                else:
                    rawPred = pred
                    if ConstantsRWalk.entType:
                        type1_raw = self.entToType[e1]
                    else:
                        type1_raw = "thing"

                for i in range(2,len(ss),2):
                    e2 = ss[i]
                    if e2=="":
                        continue
                    if ConstantsRWalk.ptyped:
                        if ConstantsRWalk.check2ndArgType:
                            candType = self.entToType[e2]
                            if (not reverse and candType!=type2_raw) or (reverse and candType!=type1_raw):
                                # print ("type not match, continue: "+ pred+ str(reverse)+  e1 + e2 + candType)
                                continue

                    else:
                        if ConstantsRWalk.entType:
                            type2_raw = self.entToType[e2]
                        else:
                            type2_raw = "thing"


                    type1 = type1_raw
                    type2 = type2_raw

                    if type1_raw != type2_raw:#because we don't need visit#person#location_reverse
                        if reverse:
                            continue
                    else:
                        type1 += "_1"
                        type2 += "_2"

                    try:

                        if not ConstantsRWalk.onlyFreq:
                            prob = np.float(ss[i + 1])
                        else:
                            prob = 1
                        if prob <= ConstantsRWalk.threshold_read_prob:
                            continue # if it's sorted, you can even break
                        if (prob == 0):
                            print ("prob 0: ", ss[i], " ", ss[i + 1])
                    except Exception as e:
                        print (e.message)
                        print ("except: ", ss[i], " ", ss[i + 1])
                        break

                    if not reverse:
                        triple = e1 + "#" + pred + "#" + e2
                        entPair = e1 + "#" + e2
                        entPair_reverse = e2 + "#" + e1#not necessary for type1!=type2
                    else:
                        triple = e2 + "#" + pred + "#" + e1
                        entPair = e2 + "#" + e1
                        entPair_reverse = e1 + "#" + e2#not necessary for type1!=type2



                    if triple in self.allTriples or N_ap < ConstantsRWalk.convEArgPairNeighs:
                        count = 1

                        if triple not in ret:
                            ret[triple] = prob

                        if triple not in self.allTriples:
                            N_ap += 1
                        else:
                            count = self.allTriples[triple]
                            self.numAllSeenEdges +=1

                        pred_orig = rawPred + "#" + type1 + "#"+type2
                        pred_reverse = rawPred + "#" + type2 + "#"+type1

                        self.numAllStoredTriples += 1
                        if self.numAllStoredTriples % 100000 == 0:
                            print (self.numAllStoredTriples)
                            print ("triple to score size: ", len(ret))

                        this_types = self.get_orderedType(type1_raw, type2_raw)

                        if type1_raw == type2_raw:
                            self.add_triple_to_graph(pred_orig,entPair,prob,this_types, count)
                            self.add_triple_to_graph(pred_reverse, entPair_reverse, prob, this_types, count)
                        else:
                            if this_types == type1+"#"+type2:
                                self.add_triple_to_graph(pred_orig, entPair, prob, this_types, count)
                            else:
                                self.add_triple_to_graph(pred_orig, entPair_reverse, prob, this_types, count)

            except Exception as e:
                print ("exception:",e)
                continue

        print (str(len(ret)) + " triple scores loaded")
        return ret


    def add_triple_to_graph(self, pred, entPair, prob, this_types, count):
        if this_types not in self.typesToGraphs:
            self.typesToGraphs[this_types] = RandWalkMatrix(this_types, self)

        randWalkMat = self.typesToGraphs[this_types]
        randWalkMat.add_triple(pred,entPair,prob, count)


    #typed version
    def getScore(self, pred, featName,types):
        ss = pred.split("#")
        args = featName.split("#")
        ss_types = types.split("#")

        if (not ConstantsRWalk.unary and (len(ss) != 3 or len(args) != 2)) or \
                (ConstantsRWalk.unary and ((len(ss) != 2 or ((not ConstantsRWalk.timeStamp and len(args) != 1) or (ConstantsRWalk.timeStamp and len(args)!=2) ) ))):
            return 0

        if not ConstantsRWalk.unary:

            swap = False
            if ss_types[0]==ss_types[1]:
                if ss[1].endswith("_2") and ss[2].endswith("_1"):
                    swap = True
            else:
                if ss_types[0]!=ss[1]:
                    swap = True

            if swap:
                tmp = args[0]
                args[0] = args[1]
                args[1] = tmp
                # in same type scenario, the pred should be reversed as well:
                #e.g., us (south.of)#loc1#loc2 canada = canada (south.of)#loc2_loc1 us, so we should again look for the first
                #but, canada (exports)#loc#thing goods, in thing#loc graph, will be (exports)#loc#thing goods,canada. So,
                #it should become canada (exports)#loc#thing goods... so, only args will swap, but not the pred
                if ss_types[0]==ss_types[1]:
                    pred = ss[0]+"#"+ss[2]+"#"+ss[1]


        if not ConstantsRWalk.ptyped:
            my_pred = ss[0]
        else:
            my_pred = pred

        if not ConstantsRWalk.unary:
            triple = args[0] + "#" + my_pred + "#" + args[1]
        else:
            triple = my_pred + "#" + featName

        if triple in self.tripleToScore:
            ret = self.tripleToScore[triple]
            if triple in self.allTriples:
                ret *= self.allTriples[triple]
            return ret
        else:
            print ("triple not found: ", pred, featName, types)
            print (triple)

        return 0

    def getCosPreds(self, rel1, rel2):
        if rel1==rel2:
            return 1

        ss1 = rel1.split("#")
        ss2 = rel2.split("#")

        if not ConstantsRWalk.ptyped:
            rel1 = ss1[0]
            rel2 = ss2[0]

        if ss1[1].endswith("_2"):
            if ConstantsRWalk.ptyped:
                rel1 = ss1[0]+"#"+ss1[2]+"#"+ss1[1]+"_reverse"
            else:
                rel1 += "_reverse"

        if ss2[1].endswith("_2"):
            if ConstantsRWalk.ptyped:
                rel2 = ss2[0] + "#" + ss2[2] + "#" + ss2[1] + "_reverse"
            else:
                rel2 += "_reverse"
        try:
            r1Emb = self.relsToEmbed[rel1]
            r2Emb = self.relsToEmbed[rel2]
            ret = np.max((cosine_similarity(r1Emb, r2Emb)[0,0], 0))
            return ret
        except Exception as e:
            return 0

parser = argparse.ArgumentParser()
parser.add_argument("--probs_file_path", type=str, default="NS_probs_all.txt", nargs="?",
                help="triple probabilities path.")
parser.add_argument("--triples_path", type=str, default="convE/data/NS/all.txt", nargs="?",
                help="triple probabilities path.")
parser.add_argument("--max_new_args", type=int, default=50, nargs="?",
                help="Number of new added second (first) arguments given fixed first (second) arguments.")
parser.add_argument("--entgraph_path", type=str, default="typedEntGrDir_NS_all", nargs="?",
                help="Entailment graphs directory.")

args = parser.parse_args()

ConstantsRWalk.triples2scoresPath = args.probs_file_path
ConstantsRWalk.allTriplesPath = args.triples_path
ConstantsRWalk.convEArgPairNeighs = args.max_new_args
ConstantsRWalk.simsFolder = args.entgraph_path

rwMatFactory = RandWalkMatrixFactory()
rwMatFactory.performWalks()

