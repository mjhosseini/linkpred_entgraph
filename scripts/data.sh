root=convE/data
cd $root
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
