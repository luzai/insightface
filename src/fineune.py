from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
import lz

lz.init_mxnet()
import sys
import math
import random
import logging
import pickle
import numpy as np
from image_iter import FaceImageIter
from image_iter import FaceImageIterList
import mxnet as mx
from mxnet import sym, gluon, nd
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
from six.moves import xrange

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image

sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import finception_resnet_v2
import fmobilenet
import fmobilenetv2
import fmobilefacenet
import fxception
import fdensenet
import fdpn
import fnasnet
import spherenet
import verification
import sklearn

# sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
# import center_loss


logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None


class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0
    
    def update(self, labels, preds):
        self.count += 1
        label = labels[0]
        pred_label = preds[1]
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim == 2:
            label = label[:, 0]
        label = label.astype('int32').flatten()
        assert label.shape == pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
    
    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0
        gt_label = preds[-2].asnumpy()
        # print(gt_label)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--data-dir', default='', help='training set directory')
    parser.add_argument('--prefix', default='', help='directory to save model.')
    parser.add_argument('--pretrained', default='', help='pretrained model to load')
    parser.add_argument('--ckpt', type=int, default=1,
                        help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--loss-type', type=int, default=4, help='loss type')
    parser.add_argument('--verbose', type=int, default=2000,
                        help='do verification testing and model saving every verbose batches')
    parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
    parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
    parser.add_argument('--network', default='r50', help='specify network')
    parser.add_argument('--image-size', default='112,112', help='specify input image height and width')
    parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
    parser.add_argument('--version-input', type=int, default=1, help='network input config')
    parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
    parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
    parser.add_argument('--version-multiplier', type=float, default=1.0, help='filters multiplier')
    parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
    parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')  # todo
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
    parser.add_argument('--fc7-lr-mult', type=float, default=1.0, help='lr mult for fc7')
    parser.add_argument("--fc7-no-bias", default=False, action="store_true", help="fc7 no bias flag")
    parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
    parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
    parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--margin-a', type=float, default=1.0, help='')
    parser.add_argument('--margin-b', type=float, default=0.0, help='')
    parser.add_argument('--easy-margin', type=int, default=0, help='')
    parser.add_argument('--margin', type=int, default=4, help='margin for sphere')
    parser.add_argument('--beta', type=float, default=1000., help='param for sphere')
    parser.add_argument('--beta-min', type=float, default=5., help='param for sphere')
    parser.add_argument('--beta-freeze', type=int, default=0, help='param for sphere')
    parser.add_argument('--gamma', type=float, default=0.12, help='param for sphere')
    parser.add_argument('--power', type=float, default=1.0, help='param for sphere')
    parser.add_argument('--scale', type=float, default=0.9993, help='param for sphere')
    parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
    parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
    parser.add_argument('--color', type=int, default=0, help='color jittering aug')
    parser.add_argument('--images-filter', type=int, default=0, help='minimum images per identity filter')
    parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
    parser.add_argument('--ce-loss', default=False, action='store_true', help='if output ce loss')
    
    # DATA_DIR = "/share/data/glint"
    DATA_DIR = "/share/data/faces_ms1m_112x112"
    # NETWORK = "r100"
    NETWORK = "r50"
    # JOB = "-comb.glint"
    JOB = "-comb.ms1m"
    LOSSTP = "5"
    MODELDIR = "../logs/model-" + NETWORK + JOB
    if not os.path.exists(MODELDIR):
        os.mkdir(MODELDIR)
        # lz.mkdir_p(MODELDIR, delete=False)
    PREFIX = MODELDIR + '/model'
    LOGFILE = MODELDIR + '/log'
    
    parser.set_defaults(
        wd=0,
        lr=0.01,
        # lr=1e-5,
        # lr_steps='',  # init lr 1e-1, final lr 1e-4
        # fc7_lr_mult=1e4,
        data_dir=DATA_DIR,
        network=NETWORK,
        loss_type=LOSSTP,
        prefix=PREFIX,
        per_batch_size=100,
        target="lfw",  #
        # target="",
        ce_loss=True,
        margin_a=.9,
        margin_m=.4,
        margin_b=.15,
        # pretrained='../logs/model-r50-arcface-ms1m-refine-v1/model,0',
        # pretrained='../logs/model-r50-comb.glint/model,9',
        pretrained='../logs/model-r50-comb.ms1m/model,32',
        # verbose=60,
        ckpt=2,  # always save
    )
    
    args = parser.parse_args()
    return args


def get_symbol(args, arg_params, aux_params, layer_name='fc7'):
    data_shape = (args.image_channel, args.image_h, args.image_w)
    image_shape = ",".join([str(x) for x in data_shape])
    margin_symbols = []
    
    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    extra_loss = None
    _weight = mx.symbol.Variable(f"{layer_name}_weight", shape=(args.num_classes, args.emb_size),
                                 lr_mult=args.fc7_lr_mult,
                                 wd_mult=args.fc7_wd_mult)
    prefix = args.pretrained.split(',')[0]
    epoch = args.pretrained.split(',')[1]
    epoch = int(epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch=epoch)
    all_layers = sym.get_internals()
    embedding = all_layers['fc1_output']
    
    # elif args.loss_type == 5:  # combined
    s = args.margin_s
    m = args.margin_m
    assert s > 0.0
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=args.num_classes,
                                name=f'{layer_name}')
    if args.margin_a != 1.0 or args.margin_m != 0.0 or args.margin_b != 0.0:
        if args.margin_a == 1.0 and args.margin_m == 0.0:
            s_m = s * args.margin_b
            gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=s_m, off_value=0.0)
            fc7 = fc7 - gt_one_hot
        else:
            zy = mx.sym.pick(fc7, gt_label, axis=1)
            cos_t = zy / s
            t = mx.sym.arccos(cos_t)
            if args.margin_a != 1.0:
                t = t * args.margin_a
            if args.margin_m > 0.0:
                t = t + args.margin_m
            body = mx.sym.cos(t)
            if args.margin_b > 0.0:
                body = body - args.margin_b
            new_zy = body * s
            diff = new_zy - zy
            diff = mx.sym.expand_dims(diff, 1)
            gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
            body = mx.sym.broadcast_mul(gt_one_hot, diff)
            fc7 = fc7 + body
    
    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label=gt_label, name='softmax', normalization='valid')
    
    out_list.append(softmax)
    
    if args.ce_loss:
        # ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
        body = mx.symbol.SoftmaxActivation(data=fc7)
        body = mx.symbol.log(body)
        _label = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=-1.0, off_value=0.0)
        body = body * _label
        ce_loss = mx.symbol.sum(body) / args.per_batch_size
        out_list.append(mx.symbol.BlockGrad(ce_loss))
    out = mx.symbol.Group(out_list)
    return (out, arg_params, aux_params)


def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    
    if len(cvd) > 0:
        for i in xrange(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx), ctx, cvd)
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size == 0:
        args.per_batch_size = 128
    args.batch_size = args.per_batch_size * args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3
    
    os.environ['BETA'] = str(args.beta)
    data_dir_list = args.data_dir.split(',')
    assert len(data_dir_list) == 1
    data_dir = data_dir_list[0]
    path_imgrec = None
    path_imglist = None
    prop = face_image.load_property(data_dir)
    args.num_classes = prop.num_classes
    # image_size = prop.image_size
    image_size = [int(x) for x in args.image_size.split(',')]
    assert len(image_size) == 2
    assert image_size[0] == image_size[1]
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    assert (args.num_classes > 0)
    print('num_classes', args.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")
    
    if args.loss_type == 1 and args.num_classes > 20000:
        args.beta_freeze = 5000
        args.gamma = 0.06
    
    print('Called with argument:', args)
    data_shape = (args.image_channel, image_size[0], image_size[1])
    mean = None
    
    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    arg_params = None
    aux_params = None
    sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params, layer_name='glint_fc7')
    fixed_args = [n for n in sym.list_arguments() if 'fc7' in n]
    
    # sym.get_internals()
    # sym.list_arguments()
    # sym.list_auxiliary_states()
    # sym.list_inputs()
    # sym.list_outputs()
    
    # label_name = 'softmax_label'
    # label_shape = (args.batch_size,)
    arg_params['glint_fc7_weight'] = arg_params['fc7_weight'].copy()
    model = mx.mod.Module(
        context=ctx,
        symbol=sym,
        fixed_param_names=fixed_args,
    )
    val_dataiter = None
    
    train_dataiter = FaceImageIter(
        batch_size=args.batch_size,
        data_shape=data_shape,
        path_imgrec=path_imgrec,
        shuffle=True,
        rand_mirror=args.rand_mirror,
        mean=mean,
        cutoff=args.cutoff,
        color_jittering=args.color,
        images_filter=args.images_filter,
    )
    
    metric1 = AccMetric()
    eval_metrics = [mx.metric.create(metric1)]
    if args.ce_loss:
        metric2 = LossValueMetric()
        eval_metrics.append(mx.metric.create(metric2))
    
    if args.network[0] == 'r' or args.network[0] == 'y':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)  # resnet style
    elif args.network[0] == 'i' or args.network[0] == 'x':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)  # inception
    else:
        initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    # initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    _rescale = 1.0 / args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 20
    _cb = mx.callback.Speedometer(args.batch_size, som)
    
    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(data_dir, name + ".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)
    
    def ver_test(nbatch):
        results = []
        for i in xrange(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10,
                                                                               None, None)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            # print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results
    
    # ver_test( 0 )
    highest_acc = [0.0, 0.0]  # lfw and target
    # for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    
    if len(args.lr_steps) == 0:
        lr_steps = [40000, 60000, 80000]
        if args.loss_type >= 1 and args.loss_type <= 7:
            lr_steps = [100000, 140000, 160000]
        p = 512.0 / args.batch_size
        for l in xrange(len(lr_steps)):
            lr_steps[l] = int(lr_steps[l] * p)
    else:
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    
    def _batch_callback(param):
        # global global_step
        global_step[0] += 1
        mbatch = global_step[0]
        for _lr in lr_steps:
            if mbatch == args.beta_freeze + _lr:
                opt.lr *= 0.1
                print('lr change to', opt.lr)
                break
        
        _cb(param)
        if mbatch % 1000 == 0:
            print('lr-batch-epoch: lr ', opt.lr,
                  'nbatch ',param.nbatch,
                  'epoch ', param.epoch,
                  'mbatch ', mbatch,
                  'lr_step', lr_steps)
        
        if mbatch >= 0 and mbatch % args.verbose == 0:
            acc_list = ver_test(mbatch)
            save_step[0] += 1
            msave = save_step[0]
            do_save = False
            is_highest = False
            if len(acc_list) > 0:
                # lfw_score = acc_list[0]
                # if lfw_score>highest_acc[0]:
                #  highest_acc[0] = lfw_score
                #  if lfw_score>=0.998:
                #    do_save = True
                score = sum(acc_list)
                if acc_list[-1] >= highest_acc[-1]:
                    if acc_list[-1] > highest_acc[-1]:
                        is_highest = True
                    else:
                        if score >= highest_acc[0]:
                            is_highest = True
                            highest_acc[0] = score
                    highest_acc[-1] = acc_list[-1]
                    # if lfw_score>=0.99:
                    #  do_save = True
            if is_highest:
                do_save = True
            if args.ckpt == 0:
                do_save = False
            elif args.ckpt == 2:
                do_save = True
            elif args.ckpt == 3:
                msave = 1
            
            if do_save:
                print('saving', msave)
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
            
            print('[%d]Accuracy-Highest: %1.5f' % (mbatch, highest_acc[-1]))
        if mbatch <= args.beta_freeze:
            _beta = args.beta
        else:
            move = max(0, mbatch - args.beta_freeze)
            _beta = max(args.beta_min, args.beta * math.pow(1 + args.gamma * move, -1.0 * args.power))
        # print('beta', _beta)
        os.environ['BETA'] = str(_beta)
        if args.max_steps > 0 and mbatch > args.max_steps:
            sys.exit(0)
    
    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    # model.set_params(arg_params, aux_params)
    model.fit(train_dataiter,
              begin_epoch=begin_epoch,
              num_epoch=end_epoch,
              eval_data=val_dataiter,
              eval_metric=eval_metrics,
              kvstore='device',
              optimizer=opt,
              # optimizer_params   = optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=_batch_callback,
              epoch_end_callback=epoch_cb)


def main():
    # time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
