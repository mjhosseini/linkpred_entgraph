# linkpred_entgraph

Still under construction.

This is the implementation for the following paper:

Duality of Link Prediction and Entailment Graph Induction, Mohammad Javad Hosseini, Shay B. Cohen, Mark Johnson, and Mark Steedman. Association for Computational Linguistics (ACL 2019).



Step 1. Setup

clone the project.
git clone https://github.com/mjhosseini/linkpred_entgraph.git
cd linkpred_entgraph

Install requirements.

pip install scipy
pip install torch
pip install h5py
pip install future
pip install spacy
pip install sklearn
python -m spacy download en
pip install nltk
python -m nltk.downloader wordnet
python -m nltk.downloader verbnet
python -m nltk.downloader stopwords

Prepare the data:

Download the extracted binary relations from the NewsSpike corpus into convE/data:
cd convE/data
wget https://www.dropbox.com/s/l5jo5bt4ueu7q12/NS.tar.gz
tar -xvzf NS.tar.gz
cp -r NS NS_probs_train
cat NS_probs_train/valid.txt NS_probs_train/test.txt > NS_probs_train/valid1.txt
mv NS_probs_train/valid1.txt NS_probs_train/valid.txt
cp NS_probs_train/train.txt NS_probs_train/test.txt
cp -r NS NS_probs_all
cp NS_probs_all/all.txt NS_probs_all/test.txt

cd ../..
python convE/wrangle_KG.py NS
python convE/wrangle_KG.py NS_probs_train
python convE/wrangle_KG.py NS_probs_all


Step 2. Training the link prediction model

train convE model:

CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True  mode train	

Step 3: Computing triple (link) probabilities for seen and unseen triples

only for training triples

CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS_probs_train process True  mode probs probs_file_path NS_probs_train.txt

on all triples

CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS_probs_all process True  mode probs probs_file_path NS_probs_all.txt

Step 4: Building the entailment graphs

only for training triples

python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_train.txt --triples_path convE/data/NS/train.txt --max_new_args 0 --entgraph_path typedEntGrDir_NS_train_MC
python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_train.txt --triples_path convE/data/NS/train.txt --max_new_args 50 --entgraph_path typedEntGrDir_NS_train_AUG_MC

on all triples

python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_all.txt --triples_path convE/data/NS/all.txt --max_new_args 0 --entgraph_path typedEntGrDir_NS_all_MC
python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_all.txt --triples_path convE/data/NS/all.txt --max_new_args 50 --entgraph_path typedEntGrDir_NS_all_AUG_MC

Step 5: Evaluate the entailment graphs

Please refer to https://github.com/mjhosseini/entgraph_eval. Use the learned entailment graphs (typedEntGrDir_NS_all_MC or typedEntGrDir_NS_all_AUG_MC) as the gpath parameter.

Step 6: Improve link prediction with entailment graphs

using entailment graphs with the Marcov Chain model (random walk)

CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True mode test_entgraphs entgraph_path typedEntGrDir_NS_train_MC 1>lpred_detailed_output_MC.txt 2>&1 &

using entailment graphs with the Marcov Chain model (random walk) + augmentation with new scores

CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True mode test_entgraphs entgraph_path typedEntGrDir_NS_all_MC 1>lpred_detailed_output_MC.txt 2>&1 &


