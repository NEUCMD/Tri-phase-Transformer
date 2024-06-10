import argparse
import os
import torch
import utils.global_var

from exp.exp_main import Exp_Main

parser = argparse.ArgumentParser(description='[stateformer] fault prognosis')

parser.add_argument('--model', type=str, required=False, default='TimeMixer',help='model of experiment, options: [PdMformer, PdMformerA, PdMformerB, Transformer, LSTM, GRU, TCN]')

# data loader
parser.add_argument('--data', type=str, required=False, default='welding_gun_55090573052', help='data')
parser.add_argument('--root_path', type=str, default='./data/BMW/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='welding_gun_55090573052.csv', help='data file')
parser.add_argument('--target', type=str, default='W_Error', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--normalization',
                    choices={'standardization', 'minmax', 'per_sample_std', 'per_sample_minmax'},
                    default='standardization',
                    help='If specified, will apply normalization on t$\lambda$he input features of a dataset.')

# sequence define
parser.add_argument('--seq_len', type=int, default=600, help='input sequence length of Informer encoder')
parser.add_argument('--pred_len', type=int, default=1200, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=14, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=14, help='decoder input size')
parser.add_argument('--c_out', type=int, default=14, help='output size')

parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
parser.add_argument('--d_feature', type=int, default=4096, help='dimension of feature')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=4096, help='dimension of fcn')
parser.add_argument('--d_level', type=int, default=4, help='level of decomposition')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='relu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--padding', type=int, default=0, help='padding type')

parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=40, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type2',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

#prepare gpu
print('cuda.is_available:',torch.cuda.is_available())

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

utils.global_var._init()

Exp = Exp_Main

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_dlv{}_eb{}_mx{}_{}_{}'.format(args.model, args.data, args.seq_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.d_level,
                args.embed, args.mix, args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting, False)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, True)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting)

    torch.cuda.empty_cache()