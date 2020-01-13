This is the implementation of the following ACL 2019 paper:

**Duality of Link Prediction and Entailment Graph Induction**, *Mohammad Javad Hosseini, Shay B. Cohen, Mark Johnson, and Mark Steedman. Association for Computational Linguistics (ACL 2019).*

## Setup

### Cloning the project and Installing the requirements

    git clone https://github.com/mjhosseini/linkpred_entgraph.git
    cd linkpred_entgraph/
    sh requirements.sh

### Preparing the data

Download the extracted binary relations from the NewsSpike corpus into convE/data folder:
    
    sh data.sh

## Running the code

### Training the link prediction model

Train ConvE model by running:

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True epochs 80 mode train

### Computing triple (link) probabilities for seen and unseen triples

**Only on training triples:**

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS_probs_train process True  mode probs probs_file_path NS_probs_train.txt

**On all triples:**

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS_probs_all process True  mode probs probs_file_path NS_probs_all.txt

### Building the entailment graphs

**Only for training triples:**

    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_train.txt --triples_path convE/data/NS/train.txt --max_new_args 0 --entgraph_path typedEntGrDir_NS_train_MC
    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_train.txt --triples_path convE/data/NS/train.txt --max_new_args 50 --entgraph_path typedEntGrDir_NS_train_AUG_MC

**On all triples**

    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_all.txt --triples_path convE/data/NS/all.txt --max_new_args 0 --entgraph_path typedEntGrDir_NS_all_MC
    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_all.txt --triples_path convE/data/NS/all.txt --max_new_args 50 --entgraph_path typedEntGrDir_NS_all_AUG_MC

## Evaluation

### Evaluate the entailment graphs

Please refer to https://github.com/mjhosseini/entgraph_eval. Use the learned entailment graphs (typedEntGrDir_NS_all_MC or typedEntGrDir_NS_all_AUG_MC) as the gpath parameter.

### Improve link prediction with entailment graphs

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

