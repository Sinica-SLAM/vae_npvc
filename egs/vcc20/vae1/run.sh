#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
voc=PWG
stage=-1
stop_stage=100
gpu=0
ngpu=1       # number of gpu in training
nj=64        # number of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 1, get more log)
seed=777     # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=24000      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# config files
train_config=conf/train_pytorch_vqvae.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
vcc20trdir=/mnt/md0/playground/vcc2020_training
vcc20evdir=/mnt/md0/playground/vcc2020_evaluation
vctkdir=/mnt/md0/playground/VCTK-Corpus/wav

# base url for downloads.
data_url=    # It's not available online, now.

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
eval_set=test

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    # mkdir -p ${datadir}
    # for part in dev-clean test-clean train-clean-100 train-clean-360; do
    #     local/download_and_untar.sh ${datadir} ${data_url} ${part}
    # done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/generate_vctk_list.py --data_root $vctkdir --list_dir data --fs 24000
    local/generate_vcc2020_traindev_list.py --data_root $vcc20trdir --list_dir data
    local/generate_vcc2020_eval_list.py --data_root $vcc20evdir --list_dir data
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    utils/combine_data.sh data/${train_set} data/vctk_train data/vcc2020_train
    utils/combine_data.sh data/${dev_set} data/vctk_dev data/vcc2020_dev
    utils/combine_data.sh data/${eval_set} data/vcc2020_test

    fbankdir=fbank
    for x in ${dev_set} ${train_set} ${eval_set}; do
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${x} \
            exp/make_fbank/${x} \
            ${fbankdir}
    done

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Speaker and Train/Valid/Eval Data Preparation"
    make_spk_id.py data/${train_set}
    make_spk_id.py data/${dev_set} --spk2spk_id data/${train_set}/spk2spk_id

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${dev_set} ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${eval_set} ${feat_ev_dir}

    # dump spk_id data
    cp data/${train_set}/utt2spk_id ${feat_tr_dir}
    cp data/${dev_set}/utt2spk_id ${feat_dt_dir}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: VQVAE model training" 
    train.py \
        --gpu ${gpu} \
        --output_dir ${expdir}/result \
        --train_dir ${feat_tr_dir} \
        --valid_dir ${feat_dt_dir} \
        --config ${train_config}
fi

outdir=${expdir}/outputs_${model}_decode
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/feats.scp ${outdir}/${name}
        # decode in parallel
        vc_decode.py \
            --gpu ${gpu} \
            --out ${outdir}/${name}/feats \
            --model ${expdir}/results/${model} \
            --config ${train_config}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis"
    pids=() # initialize pids
    # for name in ${pair_dev_set} ${pair_eval_set}; do
    for name in ${pair_eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true data/${train_set}/cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp

        # GL
        if [ ${voc} = "GL" ]; then
            echo "Using Griffin-Lim phase recovery."
            convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                --iters ${griffin_lim_iters} \
                ${outdir}_denorm/${name} \
                ${outdir}_denorm/${name}/log \
                ${outdir}_denorm/${name}/wav
        # PWG
        elif [ ${voc} = "PWG" ] || [ ${voc} = "MG" ]; then
            if [ ${voc} = "PWG" ]; then
                echo "Using Parallel WaveGAN vocoder."
                voc_expdir=exp/parallel_wavegan
            else
                echo "Using Multi-Band MelGAN vocoder."
                voc_expdir=exp/multiband_melgan
            fi
            # check existence
            if [ ! -d ${voc_expdir} ]; then
                echo "${voc_expdir} does not exist. Please download the pretrained model."
                exit 1
            fi

            # variable settings
            voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
            voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
            voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
            wav_dir=${outdir}_denorm/${name}/pwg_wav
            hdf5_norm_dir=${outdir}_denorm/${name}/hdf5_norm
            [ ! -e "${wav_dir}" ] && mkdir -p ${wav_dir}
            [ ! -e ${hdf5_norm_dir} ] && mkdir -p ${hdf5_norm_dir}

            # normalize and dump them
            echo "Normalizing..."
            ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
                parallel-wavegan-normalize \
                    --skip-wav-copy \
                    --config "${voc_conf}" \
                    --stats "${voc_stats}" \
                    --feats-scp "${outdir}_denorm/${name}/feats.scp" \
                    --dumpdir ${hdf5_norm_dir} \
                    --verbose "${verbose}"
            echo "successfully finished normalization."

            # decoding
            echo "Decoding start. See the progress via ${wav_dir}/decode.log."
            ${cuda_cmd} --gpu $gpu "${wav_dir}/decode.log" \
                parallel-wavegan-decode \
                    --dumpdir ${hdf5_norm_dir} \
                    --checkpoint "${voc_checkpoint}" \
                    --outdir ${wav_dir} \
                    --verbose "${verbose}"

            # renaming
            rename -f "s/_gen//g" ${wav_dir}/*.wav

            echo "successfully finished decoding."
        else
            echo "Vocoder type not supported. Only GL and PWG are available."
        fi    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Objective Evaluation"
    # for name in ${pair_dev_set} ${pair_eval_set}; do
    for name in ${pair_eval_set}; do
        local/ob_eval/evaluate.sh --nj ${nj} \
            --db_root ${vcc20gtdir} \
            --vocoder ${voc} \
            ${outdir} ${name}
        local/ob_eval/evaluate_similarity.sh --nj ${nj} \
            --db_root ${vcc20gtdir} \
            --vocoder ${voc} \
            ${outdir} ${name}        
    done
fi
