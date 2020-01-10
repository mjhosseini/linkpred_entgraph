# Modified version of the code at https://github.com/TimDettmers/ConvE

import sys
sys.path.append("..")
sys.path.append(".")
import torch.backends.cudnn as cudnn

from convE.evaluation import ranking_and_hits, ranking_and_hits_entGraph, compute_probs
from convE.model import ConvE, DistMult, Complex
from convE.util import *

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
np.set_printoptions(threshold=sys.maxsize)


timer = CUDATimer()
cudnn.benchmark = True

# parse console parameters and set global variables
Config.backend = Backends.TORCH
Config.parse_argv(sys.argv)

Config.cuda = True
Config.embedding_dim = 200

path_root = "convE/"

model_name = '{2}_{0}_{1}'.format(Config.input_dropout, Config.dropout, Config.model_name)
epochs = Config.epochs

if Config.mode == 'train':
    load = Config.load
    save = True
    test = False
    computeAllProbs = False
    testEntGraph = False
else:
    load = True
    save = False
    test = True
    if Config.mode == 'probs':
        computeAllProbs = True
        testEntGraph = False
    elif Config.mode == 'test':
        computeAllProbs = False
        testEntGraph = False
    elif Config.mode == 'test_entgraphs':
        computeAllProbs = False
        testEntGraph = True

if testEntGraph:
    gpath = Config.entgraph_path
    isCCG = True
    lower = True
    f_post_fix = "_sim.txt"
    featIdx = 0

model_path = path_root + 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)
model_path = model_path.replace("_probs_train", "").replace("_probs_all", "")

if not test:
    testEntGraph = False
if Config.dataset is None:
    Config.dataset = 'FB15k-237'

''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    full_path = path_root + 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = path_root + 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = path_root + 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = path_root + 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(Config.dataset, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()


    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)

def main():
    if Config.process: preprocess(Config.dataset, delete_data=True)
    train_triples_path = path_root + 'data/{0}/train.txt'.format(Config.dataset)
    # dev_triples_path = 'data/{0}/valid.txt'.format(Config.dataset)  # used for development
    test_triples_path = path_root + 'data/{0}/test.txt'.format(Config.dataset)

    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']
    num_entities = vocab['e1'].num_token
    train_batcher = StreamBatcher(Config.dataset, 'train', Config.batch_size, randomize=True, keys=input_keys)
    dev_rank_batcher = StreamBatcher(Config.dataset, 'dev_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)
    test_rank_batcher = StreamBatcher(Config.dataset, 'test_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)

    allRels = get_AllRels(vocab)
    allEntTokens = get_AllEntities(vocab)

    if Config.model_name is None:
        model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ConvE':

        if not test:
            model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token, allEntTokens, allRels)
        else:
            if testEntGraph:
                types2E, types2rel2idx, types2rels = read_graphs(gpath, f_post_fix, featIdx, isCCG, lower)
                model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token, allEntTokens, allRels, types2E, types2rel2idx, types2rels)
            else:
                model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token, allEntTokens, allRels)
    elif Config.model_name == 'DistMult':
        model = DistMult(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ComplEx':
        model = Complex(vocab['e1'].num_token, vocab['rel'].num_token)
    else:
        print ('Unknown model: {0}', Config.model_name)
        raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))

    eta = ETAHook('train', print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=100))

    if Config.cuda:
        model.cuda()
    if load:
        model_params = torch.load(model_path)

        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        if test:
            if computeAllProbs:
                fout_probs = open(Config.probs_file_path, 'w')
                model.eval()
                with torch.no_grad():
                    compute_probs(model, test_rank_batcher, vocab, 'test_probs',fout_probs,test_triples_path)
            else:
                model.eval()
                with torch.no_grad():
                    if testEntGraph:
                        # ranking_and_hits_entGraph(model, dev_rank_batcher, vocab, relW2idx, Config.model_name, 'dev_evaluation',train_triples_path, 20)
                        ranking_and_hits_entGraph(model, test_rank_batcher, vocab, 'test_evaluation', train_triples_path, 20)
                    else:
                        # ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', train_triples_path, 20)
                        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation', train_triples_path, 20)
            return

    else:
        model.init()

    params = [value.numel() for value in model.parameters()]

    print(params)
    print(np.sum(params))

    opt = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
    for epoch in range(epochs):
        model.train()
        for i, str2var in enumerate(train_batcher):
            opt.zero_grad()
            e1 = str2var['e1']
            rel = str2var['rel']
            e2_multi = str2var['e2_multi1_binary'].float()
            # label smoothing
            e2_multi = ((1.0-Config.label_smoothing_epsilon)*e2_multi) + (1.0/e2_multi.size(1))
            pred = model.forward(e1, rel)
            loss = model.loss(pred, e2_multi)
            loss.backward()
            opt.step()

            train_batcher.state.loss = loss.cpu()

        if save:
            print('saving to {0}'.format(model_path))
            if not os.path.isdir(path_root + 'saved_models'):
                os.mkdir(path_root + 'saved_models')
            torch.save(model.state_dict(), model_path)

        model.eval()

        if epoch%5==0:
            with torch.no_grad():
                ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation', train_triples_path, 20)
                if epoch % 10 == 0:#This was 10
                    if epoch > 0:
                        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation',  train_triples_path, 20)
                        # Let's write the rel embeddings!

        if model_name == "ConvE":
            fout = open('ents2emb_tmp_' + Config.model_name + '_' + Config.dataset + '.txt', 'w')
            lookup_tensor = torch.tensor([i for i in range(vocab['e1'].num_token)], dtype=torch.long).to('cuda')
            emb_e = model.emb_e(lookup_tensor).cpu().detach().numpy()
            for i in range(vocab['e1'].num_token):
                fout.write(vocab['e1'].idx2token[i] + '\t' + str(emb_e[i]) + '\n')

            fout.close()

            fout = open('rels2emb_'+Config.model_name+'_'+Config.dataset+'_tmp.txt', 'w')
            for i in range(vocab['rel'].num_token):
                if i in model.relIdx2Embed:
                    fout.write(vocab['rel'].idx2token[i] + '\t' + str(model.relIdx2Embed[i]) + '\n')
            fout.close()

    if model_name == "ConvE":
        #Let's write the final rel embeddings!
        fout = open('rels2emb_'+Config.model_name+'_'+Config.dataset+'.txt', 'w')
        for i in range(vocab['rel'].num_token):
            if i in model.relIdx2Embed:
                fout.write(vocab['rel'].idx2token[i] + '\t' + str(model.relIdx2Embed[i]) + '\n')
            else:
                print ("doesn't have: ", vocab['rel'].idx2token[i])


if __name__ == '__main__':
    main()
