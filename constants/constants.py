class ConstantsRWalk:
    convEArgPairNeighs = 50
    triples2scoresPath = "NS_probs_all.txt"
    allTriplesPath = "convE/data/NS/all.txt"
    simsFolder = "typedEntGrDir_NS_all"

    #The below params will be fixed for ACL 2019 experiments
    embsPath = None
    writeThreshold = 1e-4
    threshold_read_prob = 1e-6
    L = 1
    useFreq = False#whether we should multiply score by count. False for ACL experiments
    onlyFreq = False#i.e., don't use lpred scores and just use 1 for all the seen triples!
    entType = True
    ptyped = True#whether the predicate itself is typed
    check2ndArgType = True  # Added on 23 Feb 2019, to make sure that the 2nd arg also had correct type!

    normalized = False
    spectral_normalize = False
    normalize_col = False

    assert not ptyped or entType
    assert not spectral_normalize or normalized
    unary = False
    timeStamp = False

class ConstantsUnaryToBinary:
    simsFolder = "typedEntGrDir_unary/" #additional experiments
    unarySimsFolder = "typedEntGrDir_unary/" #additional experiments
    allTriplesPath = "all.txt" #additional experiments
