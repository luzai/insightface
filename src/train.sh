#!/usr/bin/env bash
export PATH=$HOME/anaconda2/bin:$PATH
export CU8=/data1/share/cuda-8.0/ && export LD_LIBRARY_PATH=$CU8/extras/CUPTI/lib64/:$LD_LIBRARY_PATH  && export LD_LIBRARY_PATH=$CU8/lib64/:$LD_LIBRARY_PATH  && export PATH=$CU8/bin:$PATH

export MXNET_CPU_WORKER_NTHREADS=64
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

#DATA_DIR=/opt/jiaguo/faces_vgg_112x112
DATA_DIR=/home/xinglu/work/faces_ms1m_112x112/faces_ms1m_112x112

NETWORK=r100
#JOB=softmax1e3
JOB=dbg
LOSSTP=0
MODELDIR="../model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
CUDA_VISIBLE_DEVICES='2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type "$LOSSTP" --prefix "$PREFIX" --per-batch-size 64 #--target lfw #> "$LOGFILE" 2>&1 &

# 24G*4 = 96G 24G -- 128
# 8G 128//3 = 42

# 13:16 -- 18:20 22700  5 hours  22700 / 300 steps/min = 76
# 18:18 22740 -- 18:20 22920   90 steps /min
# ttl : 320000 =  59 hours (160000, 240000, 320000)
# 1 epoch -- 29720 batch

#Loss
#LFW CFP-FP AgeDB-30
#Softmax 99.7 91.4 95.56
