# coding: utf-8
import time
import torch.optim
from collections import OrderedDict
from attrdict import AttrDict
import pandas as pd

try:
    import cPickle as pickle
except ImportError:
    import pickle
import json
import pdb

from src.args import build_parser

from src.train_and_evaluate import *
from src.components.models import *
from src.components.contextual_embeddings import *
from src.utils.helper import *
from src.utils.logger import *
from src.utils.expressions_transfer import *

global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = './data/'
board_path = './runs/'


def read_json(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file


USE_CUDA = False


def get_new_fold(data, pairs, group):
    new_fold = []
    for item, pair, g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        new_fold.append(pair)
    return new_fold


def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a / b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1]) / 100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num


def main(config):
    is_train = False

    ''' Set seed for reproducibility'''
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    '''GPU initialization'''
    device = gpu_init_pytorch(config.gpu)

    ''' Not CV mode '''

    # Set up configs
    run_name = config.run_name
    config.log_path = os.path.join(log_folder, run_name)
    config.model_path = os.path.join(model_folder, run_name)
    config.board_path = os.path.join(board_path, run_name)
    config.outputs_path = os.path.join(outputs_folder, run_name)

    # Load vocab
    vocab1_path = os.path.join(config.model_path, 'vocab1.p')
    vocab2_path = os.path.join(config.model_path, 'vocab2.p')
    config_file = os.path.join(config.model_path, 'config.p')
    log_file = os.path.join(config.log_path, 'log.txt')

    if config.results:
        config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

    create_save_directories(config.log_path)
    create_save_directories(config.result_path)

    logger = get_logger(run_name, log_file, logging.DEBUG)

    logger.info("Check outputs_path: {}".format(os.path.abspath(config.outputs_path)))

    logger.info('Experiment Name: {}'.format(config.run_name))
    logger.debug('Created Relevant Directories')

    logger.info('Loading Data...')

    train_ls, dev_ls = load_raw_data(data_path, config.dataset, is_train)

    pairs_trained, pairs_tested, generate_nums, copy_nums = transfer_num(train_ls, dev_ls, config.challenge_disp)
    logger.info("Check outputs_path: {}".format(os.path.abspath(config.outputs_path)))

    logger.debug('Data Loaded...')
    if is_train:
        logger.debug('Number of Training Examples: {}'.format(len(pairs_trained)))
    logger.debug('Number of Testing Examples: {}'.format(len(pairs_tested)))
    logger.debug('Extra Numbers: {}'.format(generate_nums))
    logger.debug('Maximum Number of Numbers: {}'.format(copy_nums))


    logger.info('Loading Vocab File...')
    logger.info("Check outputs_path: {}".format(os.path.abspath(config.outputs_path)))

    with open(vocab1_path, 'rb') as f:
        input_lang = pickle.load(f)
    with open(vocab2_path, 'rb') as f:
        output_lang = pickle.load(f)

    logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, input_lang.n_words))

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(config, logger, pairs_trained, pairs_tested,
                                                                    config.trim_threshold, generate_nums, copy_nums,
                                                                    input_lang, output_lang, tree=True)
    logger.info("Check outputs_path: {}".format(os.path.abspath(config.outputs_path)))

    checkpoint = get_latest_checkpoint(config.model_path, logger)
    logger.info("Check outputs_path: {}".format(os.path.abspath(config.outputs_path)))

    # When test only
    gpu = config.gpu
    mode = config.mode
    dataset = config.dataset
    batch_size = config.batch_size
    old_run_name = config.run_name
    with open(config_file, 'rb') as f:
        config = AttrDict(pickle.load(f))
        config.gpu = gpu
        config.mode = mode
        config.dataset = dataset
        config.batch_size = batch_size

    logger.info('Initializing Models...')
    logger.info("Check outputs_path: {}".format(os.path.abspath(config.outputs_path)))

    # Initialize models
    embedding = None
    if config.embedding == 'bert':
        embedding = BertEncoder(config.emb_name, device, config.freeze_emb)
    elif config.embedding == 'roberta':
        embedding = RobertaEncoder(config.emb_name, device, config.freeze_emb)
    else:
        embedding = Embedding(config, input_lang, input_size=input_lang.n_words,
                              embedding_size=config.embedding_size, dropout=config.dropout)
    logger.info("Check outputs_path: {}".format(os.path.abspath(config.outputs_path)))

    # encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=config.embedding_size, hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
    encoder = EncoderSeq(cell_type=config.cell_type, embedding_size=config.embedding_size,
                         hidden_size=config.hidden_size, n_layers=config.depth, dropout=config.dropout)
    predict = Prediction(hidden_size=config.hidden_size,
                         op_nums=output_lang.n_words - config.copy_nums - 1 - config.len_generate_nums,
                         input_size=config.len_generate_nums, dropout=config.dropout)
    generate = GenerateNode(hidden_size=config.hidden_size,
                            op_nums=output_lang.n_words - config.copy_nums - 1 - config.len_generate_nums,
                            embedding_size=config.embedding_size, dropout=config.dropout)
    merge = Merge(hidden_size=config.hidden_size, embedding_size=config.embedding_size, dropout=config.dropout)
    # the embedding layer is only for generated number embeddings, operators, and paddings

    logger.debug('Models Initialized')

    epoch_offset, min_train_loss, max_train_acc, max_val_acc, equation_acc, best_epoch, generate_nums = load_checkpoint(
        config, embedding, encoder, predict, generate, merge, config.mode, checkpoint, logger, device)
    logger.info("Check outputs_path: {}".format(os.path.abspath(config.outputs_path)))

    logger.info('Prediction from')
    od = OrderedDict()
    od['epoch'] = epoch_offset
    od['min_train_loss'] = min_train_loss
    od['max_train_acc'] = max_train_acc
    od['max_val_acc'] = max_val_acc
    od['equation_acc'] = equation_acc
    od['best_epoch'] = best_epoch
    print_log(logger, od)

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])

    value_ac = 0
    equation_ac = 0
    eval_total = 0
    start = time.time()

    outputs_txt_fullpath = os.path.join(config.outputs_path, "outputs.txt")
    logger.info("Save outputs to: {}".format(os.path.abspath(outputs_txt_fullpath)))
    if not os.path.isfile(outputs_txt_fullpath):
        open(outputs_txt_fullpath, 'w')

    with open(outputs_txt_fullpath, 'a') as f_out:
        f_out.write('---------------------------------------\n')
        f_out.write('Test Name: ' + old_run_name + '\n')
        f_out.write('---------------------------------------\n')
        f_out.close()

    test_res_ques, test_res_act, test_res_gen, test_res_scores = [], [], [], []

    ex_num = 0
    for test_batch in test_pairs:
        batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4],
                                               test_batch[5])
        test_res = evaluate_tree(config, test_batch[0], test_batch[1], generate_num_ids, embedding, encoder,
                                 predict, generate,
                                 merge, input_lang, output_lang, test_batch[4], test_batch[5], batch_graph,
                                 test_batch[7], beam_size=config.beam_size)
        val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4],
                                                          test_batch[6])

        cur_result = 0
        if val_ac:
            value_ac += 1
            cur_result = 1
        if equ_ac:
            equation_ac += 1
        eval_total += 1

        with open(outputs_txt_fullpath, 'a') as f_out:
            f_out.write('Example: ' + str(ex_num) + '\n')
            f_out.write('Source: ' + stack_to_string(sentence_from_indexes(input_lang, test_batch[0])) + '\n')
            f_out.write('Target: ' + stack_to_string(sentence_from_indexes(output_lang, test_batch[2])) + '\n')
            f_out.write('Generated: ' + stack_to_string(sentence_from_indexes(output_lang, test_res)) + '\n')
            if config.nums_disp:
                src_nums = len(test_batch[4])
                tgt_nums = 0
                pred_nums = 0
                for k_tgt in sentence_from_indexes(output_lang, test_batch[2]):
                    if k_tgt not in ['+', '-', '*', '/']:
                        tgt_nums += 1
                for k_pred in sentence_from_indexes(output_lang, test_res):
                    if k_pred not in ['+', '-', '*', '/']:
                        pred_nums += 1
                f_out.write('Numbers in question: ' + str(src_nums) + '\n')
                f_out.write('Numbers in Target Equation: ' + str(tgt_nums) + '\n')
                f_out.write('Numbers in Predicted Equation: ' + str(pred_nums) + '\n')
            f_out.write('Result: ' + str(cur_result) + '\n' + '\n')
            f_out.close()

        ex_num += 1

    results_df = pd.DataFrame([test_res_ques, test_res_act, test_res_gen, test_res_scores]).transpose()
    results_df.columns = ['Question', 'Actual Equation', 'Generated Equation', 'Score']
    csv_file_path = os.path.join(config.outputs_path, config.dataset + '.csv')
    results_df.to_csv(csv_file_path, index=False)
    logger.info("sum(test_res_scores):{}".format(sum(test_res_scores)))
    logger.info("test_res_scores:{}".format(test_res_scores))

    logger.info('Accuracy: {}'.format(sum(test_res_scores) / len(test_res_scores)))


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    config = args
    print(config)
    main(config)
