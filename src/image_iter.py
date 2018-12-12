from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio

import lz

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
# import face_preprocess
# import multiprocessing

logger = logging.getLogger()


class FaceImageIter(io.DataIter):
    def __init__(self, batch_size, data_shape,
                 path_imgrec=None,
                 shuffle=False, aug_list=None, mean=None,
                 rand_mirror=False, cutoff=0, color_jittering=0,
                 images_filter=0,
                 data_name='data', label_name='softmax_label',
                 metric_learning=True,
                 # metric_learning=False, # todo  metric learning batch formation method do not affetcts the scale of loss ?
                 **kwargs):
        super(FaceImageIter, self).__init__()
        self.metric_learning = metric_learning
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4] + ".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec,
                                                     'r')  # pylint: disable=redefined-variable-type
            try:
                self.imgidx, self.seq_identity, self.id2range = lz.msgpack_load(
                    os.path.dirname(path_imgidx) + '/info.mxnet.pk', use_list=True)
            except:
                s = self.imgrec.read_idx(0)
                header, _ = recordio.unpack(s)
                if header.flag > 0:
                    print('header0 label', header.label)
                    self.header0 = (int(header.label[0]), int(header.label[1]))
                    # assert(header.flag==1)
                    # self.imgidx = list(range(1, int(header.label[0])))
                    self.imgidx = []
                    self.id2range = {}
                    self.seq_identity = list(range(int(header.label[0]), int(header.label[1])))
                    for identity in self.seq_identity:
                        s = self.imgrec.read_idx(identity)
                        header, _ = recordio.unpack(s)
                        a, b = int(header.label[0]), int(header.label[1])
                        count = b - a
                        if count < images_filter:
                            continue
                        self.id2range[identity] = (a, b)
                        self.imgidx += range(a, b)
                else:
                    self.imgidx = list(self.imgrec.keys)
                lz.msgpack_dump([self.imgidx, self.seq_identity, self.id2range],
                                os.path.dirname(path_imgidx) + '/info.mxnet.pk')
            print('id2range', len(self.id2range))
            if shuffle:
                self.seq = self.imgidx
                self.oseq = self.imgidx
                print(len(self.seq))
            else:
                self.seq = None
        
        self.mean = mean
        self.nd_mean = None
        if self.mean:
            self.mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
            self.nd_mean = mx.nd.array(self.mean).reshape((1, 1, 3))
        
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d' % (data_shape[1], data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cutoff = cutoff
        self.color_jittering = color_jittering
        self.CJA = mx.image.ColorJitterAug(0.125, 0.125, 0.125)
        self.provide_label = [(label_name, (batch_size,))]
        logging.info(f'one iter: provide {self.provide_data}, {self.provide_label}')
        # print(self.provide_label[0][1])
        self.cur = 0
        self.nbatch = 0
        self.is_init = False
        self.num_instances = 4
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.inds_queue = []
    
    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.seq)
        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()
    
    def num_samples(self):
        return len(self.seq)
    
    def get_batch_ids(self):
        pids = []
        # dop = gl_conf.dop
        # lz.logging.info(f'dop smapler {np.count_nonzero( dop == -1 )} {dop}')
        pids_now = np.random.choice(self.seq_identity,
                                    size=int(self.num_pids_per_batch),
                                    replace=False)
        pids.extend(pids_now.tolist())
        # while len(pids) < self.num_pids_per_batch:
        #     pids_next = []
        #     for pid in pids_now:
        #         if dop[pid] == -1 or dop[pid] in pids_next or dop[pid] in pids:
        #             pid_t = np.random.choice(self.ids, )
        #             # make sure id is unique
        #             while pid_t in pids_next or pid_t in pids:
        #                 pid_t = np.random.choice(self.ids, )
        #             pids_next.append(pid_t)
        #         else:
        #             pids_next.append(dop[pid])
        #     pids.extend(pids_next)
        #     pids_now = pids_next
        assert len(pids) == np.unique(pids).shape[0]
        return pids
    
    def get_batch_idxs(self):
        inds = []
        pids = self.get_batch_ids()
        for pid in pids:
            a, b = self.id2range[pid]
            inds.extend(
                np.random.choice(
                    list(range(a, b)),
                    size=(self.num_instances,),
                    replace=True,
                ).tolist()
            )
        inds = inds[:self.batch_size]
        return inds
    
    def next_sample_metric_learning(self):
        while True:
            if self.cur >= len(self.seq):
                raise StopIteration
            # if self.cur % self.batch_size == 0:
            if len(self.inds_queue) == 0:
                self.inds_queue += self.get_batch_idxs()
            idx = self.inds_queue.pop(0)
            # print(idx, len(self.inds_queue), self.inds_queue[:10], )
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                label = header.label
                if not isinstance(label, numbers.Number):
                    label = label[0]
                return label, img, None, None
            else:
                label, fname, bbox, landmark = self.imglist[idx]
                return label, self.read_image(fname), bbox, landmark
    
    def next_sample(self):
        """Helper function for reading in next sample."""
        # set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
            while True:
                if self.cur >= len(self.seq):
                    raise StopIteration
                idx = self.seq[self.cur]
                self.cur += 1
                if self.imgrec is not None:
                    s = self.imgrec.read_idx(idx)
                    header, img = recordio.unpack(s)
                    label = header.label
                    if not isinstance(label, numbers.Number):
                        label = label[0]
                    return label, img, None, None
                else:
                    label, fname, bbox, landmark = self.imglist[idx]
                    return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None
    
    def brightness_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        src *= alpha
        return src
    
    def contrast_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = nd.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
        src *= alpha
        src += gray
        return src
    
    def saturation_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = nd.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = nd.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        src *= alpha
        src += gray
        return src
    
    def color_aug(self, img, x):
        # augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
        # random.shuffle(augs)
        # for aug in augs:
        # #print(img.shape)
        #  img = aug(img, x)
        # #print(img.shape)
        # return img
        return self.CJA(img)
    
    def mirror_aug(self, img):
        _rd = random.randint(0, 1)
        if _rd == 1:
            for c in range(img.shape[2]):
                img[:, :, c] = np.fliplr(img[:, :, c])
        return img
    
    def compress_aug(self, img):
        buf = BytesIO()
        img = Image.fromarray(img.asnumpy(), 'RGB')
        q = random.randint(2, 20)
        img.save(buf, format='JPEG', quality=q)
        buf = buf.getvalue()
        img = Image.open(BytesIO(buf))
        return nd.array(np.asarray(img, 'float32'))
    
    __next__ = next
    
    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        """Returns the next batch of data."""
        # print('in next', self.cur, self.labelcur)
        self.nbatch += 1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
            batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                if self.metric_learning:
                    label, s, bbox, landmark = self.next_sample_metric_learning()
                else:
                    label, s, bbox, landmark = self.next_sample()
                _data = self.imdecode(s)
                if _data.shape[0] != self.data_shape[1]:
                    _data = mx.image.resize_short(_data, self.data_shape[1])
                if self.rand_mirror:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        _data = mx.ndarray.flip(data=_data, axis=1)
                if self.color_jittering > 0:
                    if self.color_jittering > 1:
                        _rd = random.randint(0, 1)
                        if _rd == 1:
                            _data = self.compress_aug(_data)
                    # print('do color aug')
                    _data = _data.astype('float32', copy=False)
                    # print(_data.__class__)
                    _data = self.color_aug(_data, 0.125)
                if self.nd_mean is not None:
                    _data = _data.astype('float32', copy=False)
                    _data -= self.nd_mean
                    _data *= 0.0078125
                if self.cutoff > 0:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        # print('do cutoff aug', self.cutoff)
                        centerh = random.randint(0, _data.shape[0] - 1)
                        centerw = random.randint(0, _data.shape[1] - 1)
                        half = self.cutoff // 2
                        starth = max(0, centerh - half)
                        endh = min(_data.shape[0], centerh + half)
                        startw = max(0, centerw - half)
                        endw = min(_data.shape[1], centerw + half)
                        
                        # print(starth, endh, startw, endw, _data.shape)
                        _data[starth:endh, startw:endw, :] = 128
                data = [_data]  # here data is RGB format
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                # print('aa',data[0].shape)
                # data = self.augmentation_transform(data)
                # print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    # print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i < batch_size:
                raise StopIteration
        
        return io.DataBatch([batch_data], [batch_label], batch_size - i)
    
    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')
    
    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')
    
    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)  # mx.ndarray
        return img
    
    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.
    
        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img
    
    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data
    
    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))


class FaceImageIterList(io.DataIter):
    def __init__(self, iter_list):
        assert len(iter_list) > 0
        self.provide_data = iter_list[0].provide_data
        self.provide_label = iter_list[0].provide_label
        self.iter_list = iter_list
        self.cur_iter = None
    
    def reset(self):
        self.cur_iter.reset()
    
    def next(self):
        self.cur_iter = random.choice(self.iter_list)
        while True:
            try:
                ret = self.cur_iter.next()
            except StopIteration:
                self.cur_iter.reset()
                continue
            return ret


'''
import torch
from pathlib import Path
from torch.utils.data import DataLoader


class Dataset():
    def __init__(self, train_iter, batch_size=100):
        self.train_iter = train_iter
        self.train_iter_back = iter(train_iter)
        self.batch_size = batch_size
        self.num_classes = 85164
        self.root = Path('/data1/share/faces_ms1m_112x112/train.rec')
        assert self.root.exists()
    
    def __len__(self):
        # 85k id, 8M imgs
        # return len(self.train_iter)
        return int(8e6 / self.batch_size) + 1  # 8e4
    
    def __getitem__(self, indices, ):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        res = self._get_single_item(indices)
        for k, v in res.items():
            assert (
                    isinstance(v, np.ndarray) or
                    isinstance(v, str) or
                    isinstance(v, int) or
                    isinstance(v, np.int64) or
                    torch.is_tensor(v)
            ), type(v)
        return res
    
    def _get_single_item(self, index):
        try:
            next_data_batch = next(self.train_iter)
        except StopIteration:
            print('this batch finish ')
            self.train_iter = iter(self.train_iter_back)
            next_data_batch = next(self.train_iter)
        imgs = next_data_batch.data[0].asnumpy()
        labels = next_data_batch.label[0].asnumpy()
        return {'imgs': imgs, 'labels': labels}


class Loader2(object):
    def __init__(self, train_dataiter, bs):
        self.dataset = Dataset(train_dataiter, batch_size=bs)
        train_loader = DataLoader(
            self.dataset, batch_size=1, num_workers=0,
        )
        
        self.train_loader = train_loader
        self.ind = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ind < len(self):
            for val in self.train_loader:
                imgs, labels = val['imgs'], val['labels']
                imgs = torch.squeeze(imgs).cuda()
                labels = torch.squeeze(labels).long().cuda()
                return imgs, labels
        else:
            raise StopIteration()
    
    def __len__(self):
        return len(self.train_loader)


if __name__ == '__main__':
    import os
    
    os.environ['MXNET_CPU_WORKER_NTHREADS'] = "12"
    os.environ['MXNET_ENGINE_TYPE'] = "ThreadedEnginePerDevice"
    
    bs = 4
    train_dataiter = FaceImageIter(
        batch_size=bs,
        data_shape=(3, 112, 112),
        path_imgrec='/home/xinglu/work/faces_ms1m_112x112/train.rec',
        shuffle=True,
        rand_mirror=True,
        mean=None,
        cutoff=0,
    )
    # for batch in train_dataiter:
    #     break
    # print(batch.data[0].shape, ' 1')
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)
    for ind, batch in enumerate(train_dataiter):
        # if ind % 1000 == 0:
        #     print(ind)
        break
    print(batch.label[0].asnumpy().shape)
    
    import torch
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        Dataset(train_dataiter, batch_size=bs),
        batch_size=1, num_workers=0,
    )
    for val in train_loader:
        imgs, labels = val['imgs'], val['labels']
        imgs = np.squeeze(imgs, 0)
        labels = np.squeeze(labels, 0)
        break
    print(imgs.shape)
'''
