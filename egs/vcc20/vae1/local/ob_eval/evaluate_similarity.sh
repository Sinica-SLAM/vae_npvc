#!/bin/bash

#!/bin/bash

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Evaluation script for VC


echo "$0 $*"  # Print the command line for logging
. ./path.sh

nj=1
db_root=""
vocoder=
test_kind="utt"
help_message="Usage: $0 <outdir> <subset> <srcspk> <trgspk>"

. utils/parse_options.sh

outdir=$1
set_name=$2  # <srcspk>_<trgspk>_<name> 

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

# parse srcspk, trgspk and name
srcspk=$(echo ${set_name} | awk -F"_" '{print $1}')
trgspk=$(echo ${set_name} | awk -F"_" '{print $2}')
name=$(echo ${set_name} | awk -F"_" '{print $3}')
    
# Decide wavdir depending on vocoder
if [ ! -z ${vocoder} ]; then
    # select vocoder type (GL, PWG)
    if [ ${vocoder} == "PWG" ]; then
        wavdir=${outdir}_denorm/${set_name}/pwg_wav
    elif [ ${vocoder} == "GL" ]; then
        wavdir=${outdir}_denorm/${set_name}/wav
    else
        echo "Vocoder type other than GL, PWG is not supported!"
        exit 1
    fi
else
    echo "Please specify vocoder."
    exit 1
fi

echo "step 0: Model preparation"
# Speaker model download (VoxCeleb)
spk_model_name=xvector_nnet_1a
nnet_dir=exp/${spk_model_name}
plda_dir=${nnet_dir}/xvectors_train_combined_200k
if [ ! -e ${nnet_dir} ]; then
    echo "X-vector model does not exist. Download pre-trained model."
    wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
    tar xvf 0008_sitw_v2_1a.tar.gz
    mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
    rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
fi
echo "X-vector model: ${nnet_dir} exits."

# setting dir
asv_data_dir="${outdir}_denorm.ob_eval/${spk_model_name}_asv.data"
asv_mfcc_dir="${outdir}_denorm.ob_eval/${spk_model_name}_asv.mfcc"
asv_xvec_dir="${outdir}_denorm.ob_eval/${spk_model_name}_asv.xvector"
asv_log_dir="${outdir}_denorm.ob_eval/${spk_model_name}_asv.log"
asv_result_dir="${outdir}_denorm.ob_eval/${spk_model_name}_asv.result"

echo "step 1: Data preparation for ASV"
# Data preparation for ASV
local/ob_eval/data_prep_for_asr.sh ${db_root}/${trgspk} ${asv_data_dir}/${trgspk} ${trgspk}_enroll
utils/validate_data_dir.sh --no-feats --no-text ${asv_data_dir}/${trgspk}

local/ob_eval/data_prep_for_asr.sh ${wavdir} ${asv_data_dir}/${set_name} ${trgspk}
utils/validate_data_dir.sh --no-feats --no-text ${asv_data_dir}/${set_name}

echo "step 2: Feature extraction for ASV"
# Extract mfcc, vad decision, and x-vector
mfccdir=mfcc
vaddir=mfcc
for name in ${set_name} ${trgspk}; do
    # Make MFCCs and compute the energy-based VAD for each dataset
    nj_tmp=`cat ${asv_data_dir}/${name}/spk2utt | wc -l`
    [ $nj_tmp -gt $nj ] && nj_tmp=$nj || echo "n_spk is less then n_job, reduce it from ${nj} to ${nj_tmp}"

    steps/make_mfcc.sh \
        --write-utt2num-frames true \
        --mfcc-config conf/mfcc.conf \
        --nj ${nj_tmp} --cmd "$train_cmd" \
        ${asv_data_dir}/${name} ${asv_log_dir}/${name} ${asv_mfcc_dir}/${name}
    utils/fix_data_dir.sh ${asv_data_dir}/${name}
    sid/compute_vad_decision.sh --nj ${nj_tmp} --cmd "$train_cmd" \
        ${asv_data_dir}/${name} ${asv_log_dir}/${name} ${asv_mfcc_dir}/${name}
    utils/fix_data_dir.sh ${asv_data_dir}/${name}
    # Extract x-vector
    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj_tmp} \
        ${nnet_dir} ${asv_data_dir}/${name} \
        ${asv_xvec_dir}/${name}
done

echo "step 3: Similarity scoring for ASV"
trials=${asv_xvec_dir}/${set_name}/trials

enroll_scp=${asv_xvec_dir}/${trgspk}/xvector.scp
if [ $test_kind == 'spk' ]; then
    test_scp=${asv_xvec_dir}/${set_name}/spk_xvector.scp
else
    test_scp=${asv_xvec_dir}/${set_name}/xvector.scp
fi

awk -v spk=${trgspk}_enroll '{print(spk, $1)}' ${test_scp} > ${trials}

mkdir -p ${asv_result_dir}/${trgspk}

# Calculate PLDA and cosine similarity
$train_cmd ${asv_log_dir}/${trgspk}/plda_scoring.log \
ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${asv_xvec_dir}/${trgspk}/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${plda_dir}/plda - |" \
    "ark:ivector-mean ark:${asv_data_dir}/${trgspk}/spk2utt scp:${asv_xvec_dir}/${trgspk}/xvector.scp ark:- | ark:ivector-subtract-global-mean ${plda_dir}/mean.vec ark:- ark:- | transform-vec ${plda_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${plda_dir}/mean.vec scp:${test_scp} ark:- | transform-vec ${plda_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '${trials}' | cut -d\  --fields=1,2 |" ${asv_result_dir}/${trgspk}/plda_scores || exit 1;

$train_cmd ${asv_log_dir}/${trgspk}/cossim_scoring.log \
  ivector-compute-dot-products \
    "cat '${trials}' | cut -d\  --fields=1,2 |" \
    "ark:ivector-normalize-length --scaleup=false scp:${enroll_scp} ark:- |" \
    "ark:ivector-normalize-length --scaleup=false scp:${test_scp} ark:- |" \
    ${asv_result_dir}/${trgspk}/cossim_scores

if [ $test_kind != 'spk' ]; then
    mean_plda_score=$(awk 'BEGIN{a=0}{a+=$3} END{print(a/NR)}' ${asv_result_dir}/${trgspk}/plda_scores)
    echo "${trgspk}_enroll Mean $mean_plda_score" >> ${asv_result_dir}/${trgspk}/plda_scores

    mean_cossim_score=$(awk 'BEGIN{a=0}{a+=$3} END{print(a/NR)}' ${asv_result_dir}/${trgspk}/cossim_scores)
    echo "${trgspk}_enroll Mean $mean_cossim_score" >> ${asv_result_dir}/${trgspk}/cossim_scores
fi
