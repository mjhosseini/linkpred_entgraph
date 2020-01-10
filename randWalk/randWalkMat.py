import sys
sys.path.append("..")
from constants.constants import ConstantsRWalk
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np

from collections import defaultdict

class RandWalkMatrix:

    def __init__(self, types, rwalkFact):
        self.entPairToPred = defaultdict(set)
        self.predToEntPair = defaultdict(set)
        self.entPairToSumNeighs = defaultdict(np.double)
        self.predToSumNeighs = defaultdict(np.double)
        self.pred2idx = dict()
        self.entPair2idx = dict()
        self.allPreds = []
        self.allEntPairs = []
        self.types = types
        self.rwalkFact = rwalkFact
        self.randWalkMats = []
        self.A = None
        self.B = None
        self.Au = None #none-normalized P to be used in rw training: pred by arg-pair
        self.Bu = None #none-normalized A to be used in rw training: arg-pair by pred
        self.epairEmbs = None
        self.epairWeights = None

    def init_matrix(self):

        self.numPreds = len(self.predToEntPair)
        idx = 0
        N = 0

        for p in self.predToEntPair:
            self.pred2idx[p] = idx
            self.allPreds.append(p)
            idx += 1
            N += len(self.predToEntPair[p])

        N2 = 0
        idx = 0
        for ap in self.entPairToPred:
            self.entPair2idx[ap] = idx
            self.allEntPairs.append(ap)
            idx += 1
            N2 += len(self.entPairToPred[ap])
        print (N, N2)
        assert N == N2

        print("N is: ", N ," " ,N, len(self.predToEntPair) , len(self.entPairToPred))

        #making matrices A(pred by ent-pair) and B(ent-pair by pred)

        dataA = np.zeros(N)
        dataAu = np.zeros(N)
        rowA = np.zeros(N)
        colA = np.zeros(N)
        dataB = np.zeros(N)
        dataBu = np.zeros(N)
        rowB = np.zeros(N)
        colB = np.zeros(N)
        idx = 0

        #A,B
        for pred in self.predToEntPair:
            r = self.pred2idx[pred]
            for entPair in self.predToEntPair[pred]:
                c = self.entPair2idx[entPair]
                rowA[idx] = r
                colA[idx] = c
                if not ConstantsRWalk.spectral_normalize:
                    dataA[idx] = self.rwalkFact.getScore(pred, entPair,self.types) / self.predToSumNeighs[pred]
                else:
                    dataA[idx] = self.rwalkFact.getScore(pred, entPair, self.types) / np.sqrt(self.predToSumNeighs[pred] * self.entPairToSumNeighs[entPair])
                dataAu[idx] = self.rwalkFact.getScore(pred, entPair,self.types)

                rowB[idx] = c
                colB[idx] = r
                if not ConstantsRWalk.spectral_normalize:
                    dataB[idx] = self.rwalkFact.getScore(pred, entPair, self.types) / self.entPairToSumNeighs[entPair]
                else:
                    dataB[idx] = self.rwalkFact.getScore(pred, entPair, self.types) / np.sqrt(self.predToSumNeighs[pred] * self.entPairToSumNeighs[entPair])

                dataBu[idx] = self.rwalkFact.getScore(pred, entPair,self.types)
                idx += 1


        A = csr_matrix((dataA, (rowA, colA)), shape=(len(self.predToEntPair), len(self.entPairToPred)))
        B = csr_matrix((dataB, (rowB, colB)), shape=(len(self.entPairToPred), len(self.predToEntPair)))

        self.Au = csr_matrix((dataAu, (rowA, colA)), shape=(len(self.predToEntPair), len(self.entPairToPred)))
        self.Bu = csr_matrix((dataBu, (rowB, colB)), shape=(len(self.entPairToPred), len(self.predToEntPair)))
        self.A = A
        # self.B = B

        self.mat = A*B
        # RandWalkMatrix.applyThreshold(self.mat,ConstantsRWalk.threshold)
        print ("mat size: ", self.mat.getnnz())


    #experimental code, not used in the paper (results didn't improve)!
    #normalize by both row and colum (finally row, to make sure out-degree sums to 1, but in-degree almost sums to 1!)
    def normalizeRowCol(self,mat):
        for i in range(100):
            mat = normalize(mat,norm='l1',axis=0)
            mat = normalize(mat,norm='l1',axis=1)

        print ("mat sum rows and cols:")
        for i in range(self.mat.shape[0]):
            print ("sum row ", i, ":", np.sum(self.mat[i, :].todense()))
            print ("sum col ", i, ":", np.sum(self.mat[:, i].todense()))


    def setRandWalkMat(self, K):

        if ConstantsRWalk.normalize_col:
            self.normalizeRowCol(self.mat)

        randWalkMat = self.mat
        self.randWalkMats.append(randWalkMat)


        for k in range(K-1):
            randWalkMat = randWalkMat*self.mat
            # self.applyThreshold(randWalkMat,ConstantsRWalk.threshold)
            self.randWalkMats.append(randWalkMat)

    def writeResults(self):
        fnameTProp = ConstantsRWalk.simsFolder + "/" + self.types + "_sim.txt"

        op = open(fnameTProp,'w')

        N = len(self.predToEntPair)
        op.write(self.types + " " + " num preds: " + str(N)+"\n")

        lastMat = self.randWalkMats[-1]
        nnzs = lastMat.getnnz(0)

        for predIdx, pred in enumerate(self.predToEntPair):

            op.write("predicate: " + pred+"\n")
            op.write("max num neighbors: " + str(nnzs[predIdx])+"\n")
            op.write("\n")

            for L in range(len(self.randWalkMats)):
                thisMat = self.randWalkMats[L]

                scores = []
                scores_rw_cos = []
                scores_cos = []

                sumOuts = 0

                selfProb = thisMat[predIdx,predIdx]
                thisRow = thisMat[predIdx,:]
                neighs = thisRow.nonzero()[1]
                neighVals = thisRow.data

                # print neighVals
                for i,neigh in enumerate(neighs):
                    pred2 = self.allPreds[neigh]
                    w = neighVals[i]
                    sumOuts += w

                    if ConstantsRWalk.normalized:
                        w /= selfProb
                        w = min(w,1.0)

                    if w < ConstantsRWalk.writeThreshold:
                        continue

                    ss = pred.split("#")
                    ss2 = pred2.split("#")
                    pred_unt = ss[0]+"#thing_1#thing_2"
                    if ss[1]==ss2[1]:
                        pred_unt2 = ss2[0] + "#thing_1#thing_2"
                    else:
                        pred_unt2 = ss2[0]+"#thing_2#thing_1"

                    if ConstantsRWalk.embsPath:

                        if not ConstantsRWalk.ptyped:
                            w_rw_cos = np.sqrt(w * self.rwalkFact.getCosPreds(pred_unt, pred_unt2))
                            w_cos = self.rwalkFact.getCosPreds(pred_unt, pred_unt2)
                        else:
                            w_rw_cos = np.sqrt(w * self.rwalkFact.getCosPreds(pred, pred2))
                            w_cos = self.rwalkFact.getCosPreds(pred, pred2)

                    scores.append((pred2,w))
                    if ConstantsRWalk.embsPath:
                        scores_rw_cos.append((pred2, w_rw_cos))
                        scores_cos.append((pred2, w_cos))

                if (sumOuts<.99 or sumOuts>1.01) and not ConstantsRWalk.normalized:
                    print ("pred: ", pred, L)
                    print ("sanity: " + str(sumOuts))


                scores = sorted(scores,key = lambda x: x[1],reverse=True)
                if ConstantsRWalk.embsPath:
                    scores_rw_cos = sorted(scores_rw_cos, key = lambda x: x[1],reverse=True)
                    scores_cos = sorted(scores_cos, key=lambda x: x[1], reverse=True)
                # scores_lin = sorted(scores_lin, key=lambda x: x[1], reverse=True)

                op.write("rand walk " + str(L) + " sims\n")
                for pred2,w in scores:
                    op.write(pred2 + " " + str(w)+"\n")


                op.write("\n")

                if ConstantsRWalk.embsPath:
                    op.write("rand walk cos " + str(L) + " sims\n")
                    for pred2,w_rw_cos in scores_rw_cos:
                        op.write(pred2 + " " + str(w_rw_cos)+"\n")
                    op.write("\n")

                    op.write("cos " + str(L) + " sims\n")
                    for pred2, w_cos in scores_cos:
                        op.write(pred2 + " " + str(w_cos) + "\n")
                    op.write("\n")

        op.close()

        print("results written for: ", fnameTProp)

    def add_triple(self, pred, entPair, prob, count):
        if pred not in self.entPairToPred[entPair]:
            self.entPairToPred[entPair].add(pred)
            self.entPairToSumNeighs[entPair] += count * prob

            self.predToEntPair[pred].add(entPair)
            self.predToSumNeighs[pred] += count * prob
