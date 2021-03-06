from recognition.embedding import Embedding
from lz import *
import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch

# ijb_path = '/data2/share/ijbc/'
# test1_path = ijb_path + '/IJB-C/protocols/test1/'
# img_path = ijb_path + 'IJB/IJB-C/images/'
# df_enroll = pd.read_csv(test1_path + '/enroll_templates.csv')
# df_verif = pd.read_csv(test1_path + '/verif_templates.csv')
# df_match = pd.read_csv(test1_path + '/match.csv')
# dst = ijb_path + '/ijb.test1.proc/'
#
# df1 = df_enroll[['TEMPLATE_ID', 'SUBJECT_ID']].groupby('TEMPLATE_ID').mean()
# df2 = df_verif[['TEMPLATE_ID', 'SUBJECT_ID']].groupby('TEMPLATE_ID').mean()
# df = pd.concat((df1, df2))
# t2s = dict(zip(df.index, df.SUBJECT_ID))
# all_tids = list(t2s.keys())

IJBC_path = '/data1/share/IJB_release/'
ijbcp = IJBC_path + 'ijbc.info.h5'
try:
    df_tm, df_pair, df_name = df_load(ijbcp, 'tm'), df_load(ijbcp, 'pair'), df_load(ijbcp, 'name')
except:
    fn = (os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_face_tid_mid.txt'))
    df_tm = pd.read_csv(fn, sep=' ', header=None)
    fn = (os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_template_pair_label.txt'))
    df_pair = pd.read_csv(fn, sep=' ', header=None)
    fn = os.path.join(IJBC_path + 'IJBC/meta', 'ijbc_name_5pts_score.txt')
    df_name = pd.read_csv(fn, sep=' ', header=None)
    df_dump(df_tm, ijbcp, 'tm')
    df_dump(df_pair, ijbcp, 'pair')
    df_dump(df_name, ijbcp, 'name')

confs = [
    ['logs/model-r50-arcface-ms1m-refine-v1/model', 0],
    ['logs/model-r50-comb.glint/model', 9],
    ['logs/model-r50-comb.r50.ms1m/model', 5],
    ['logs/model-r50-comb.r50.ms1m/model', 105],
    ['logs/model-r100-arcface-ms1m-refine-v2/model', 0],
    ['logs/model-r100-softmax1e3/model', 207],
]
res = []
for conf in confs:
    model_path = root_path + conf[0]
    assert os.path.exists(os.path.dirname(model_path)), os.path.dirname(model_path)
    epoch = conf[1]
    
    embedding = Embedding(model_path, epoch, 0)
    logging.info('learner loaded')
    
    # use_topk = 999
    # df_pair = df_pair.iloc[:use_topk, :]
    unique_tid = np.unique(df_pair.iloc[:, :2].values.flatten())
    import sklearn
    
    # img_feats = np.empty((df_tm.shape[0),512  ) )
    img_feats = np.ones((df_tm.shape[0], 512)) * np.nan
    for ind, row in df_name.iterrows():
        tid = df_tm.iloc[ind, 1]
        if not tid in unique_tid: continue
        imgfn = row.iloc[0]
        lmks = row.iloc[1:11]
        lmks = np.asarray(lmks, np.float32).reshape((5, 2))
        score = row.iloc[-1]
        score = float(score)
        imgfn = '/data1/share/IJB_release/IJBC/loose_crop/' + imgfn
        img = cvb.read_img(imgfn)
        fea = embedding.get(img, lmks, normalize=False)
        # fea = sklearn.preprocessing.normalize(fea )
        fea *= score
        img_feats[ind, :] = fea
    templates, medias = df_tm.values[:, 1], df_tm.values[:, 2]
    p1, p2, label = df_pair.values[:, 0], df_pair.values[:, 1], df_pair.values[:, 2]
    
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    
    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    
    from sklearn.metrics import roc_curve
    
    print(score.max(), score.min())
    _ = plt.hist(score)
    fpr, tpr, _ = roc_curve(label, score)
    
    plt.figure()
    plt.plot(fpr, tpr, '.-')
    plt.show()
    
    plt.figure()
    plt.semilogx(fpr, tpr, '.-')
    plt.show()
    
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        print(x_labels[fpr_iter], tpr[min_index])
        res.append([conf[0], conf[1], x_labels[fpr_iter], tpr[min_index], ])
    
    from sklearn.metrics import auc
    
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    
    # msgpack_dump([label, score], work_path + 'ijb3.mxnet.pk')
    # from IPython import embed
    # embed()

print(res)
