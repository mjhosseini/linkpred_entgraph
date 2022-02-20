
<div class=figure>
  <p align="center"><img src="https://www.dropbox.com/s/jz09uugbyzdkoun/toy_acl2019.jpg?raw=1"
    width="300" height=auto></p>
  <p align="center"><small><i>(A) Link Prediction, (B) Entailment Graph Induction</i></small></p>
</div>

This codbase contains the PyTorch implementation of the following paper:

**Duality of Link Prediction and Entailment Graph Induction**, *Mohammad Javad Hosseini, Shay B. Cohen, Mark Johnson, and Mark Steedman. Association for Computational Linguistics (ACL 2019).* [[paper]](https://www.aclweb.org/anthology/P19-1468.pdf)

## Setup

### Cloning the project and installing the requirements

    git clone https://github.com/mjhosseini/linkpred_entgraph.git
    cd linkpred_entgraph/
    sh scripts/requirements.sh

### Preparing the data

Download the extracted binary relations from the NewsSpike corpus into convE/data folder:
    
    sh scripts/data.sh

## Running the code

### Training the link prediction model

Train ConvE model by running:

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True epochs 80 mode train
    
Alternatively, you can copy the pre-trained model on the NewsSpike corpus:

    sh scripts/dl_pretrained.sh

### Computing triple (link) probabilities for seen and unseen triples

**Only on training triples:**

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS_probs_train process True mode probs probs_file_path NS_probs_train.txt

**On all triples:**

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS_probs_all process True mode probs probs_file_path NS_probs_all.txt

### Building the entailment graphs

Build the entailment graphs by the Marcov Chain model (random walk) as well as the Marcov Chain model (random walk) + augmentation with new scores. The former is done by --max_new_args 0 and the latter is done by --max_new_args 50. 

This step should be run on CPU, preferably with more than 100GB RAM (depending on the --max_new_args parameter).

**Only for training triples:**

    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_train.txt --triples_path convE/data/NS/train.txt --max_new_args 0 --entgraph_path typedEntGrDir_NS_train_MC
    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_train.txt --triples_path convE/data/NS/train.txt --max_new_args 50 --entgraph_path typedEntGrDir_NS_train_AUG_MC

**On all triples:**

    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_all.txt --triples_path convE/data/NS/all.txt --max_new_args 0 --entgraph_path typedEntGrDir_NS_all_MC
    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_all.txt --triples_path convE/data/NS/all.txt --max_new_args 50 --entgraph_path typedEntGrDir_NS_all_AUG_MC
    
### Learning global entailment graphs
Please refer to https://github.com/mjhosseini/entGraph/ for learning global entailment graphs from local entailment graphs (the ones that were learned above).

**We set the below parameters for the AUG models:**

*constants.ConstantsGraphs*:

* root=".../typedEntGrDir_NS_all_AUG_MC" (or ".../typedEntGrDir_NS_train_AUG_MC")
* edgeThreshold=.0002

*constants.ConstantsSoftConst*:

* lmbda=.0002, lmbda_2=1.5, and epsilon=0.3

**We set the below parameters for the non-AUG models:**

*constants.ConstantsGraphs*:

* root=".../typedEntGrDir_NS_all_MC" (or ".../typedEntGrDir_NS_train_MC")
* edgeThreshold=0

*constants.ConstantsSoftConst*:

* lmbda=0, lmbda_2=1, and epsilon=1.0


## Evaluation

### Evaluate the entailment graphs

Please refer to https://github.com/mjhosseini/entgraph_eval for evaluation.

We can use the entailment graphs that are learned by accessing all the link prediction data as here we only evaluate the entailment task, not link prediction task. Use the learned entailment graphs (typedEntGrDir_NS_all_MC or typedEntGrDir_NS_all_AUG_MC) as the gpath parameter of the entgraph_eval project.

### Improve link prediction with entailment graphs

We can use the entailment graphs that are learned by accessing only the link prediciton training data.

Using entailment graphs with the Marcov Chain model (random walk):

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True mode test_entgraphs entgraph_path typedEntGrDir_NS_train_MC 1>lpred_detailed_output_MC.txt 2>&1 &

Using entailment graphs with the Marcov Chain model (random walk) + augmentation with new scores:

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True mode test_entgraphs entgraph_path typedEntGrDir_NS_all_MC 1>lpred_detailed_output_MC.txt 2>&1 &

## Citation

If you found this codebase useful, please cite:

    @inproceedings{hosseini2019duality,
      title={Duality of Link Prediction and Entailment Graph Induction},
      author={Hosseini, Mohammad Javad and Cohen, Shay B and Johnson, Mark and Steedman, Mark},
      booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      pages={4736--4746},
      year={2019}
    }

