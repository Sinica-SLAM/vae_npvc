#!/bin/bash

./run.sh --stage 3 --stop-stage 3 --gpu 3 || exit 1;

for tgtspk in TEF1 TEF2 TEM1 TEM2; do
    echo "Converting to speaker $tgtspk ..."
    ./run.sh --stage 5 --stop-stage 7 --tgtspk $tgtspk --gpu 3 || exit 1;
done

echo "Show the result"

expdir=exp/train_pytorch_train_pytorch_vqvae/outputs_model.loss.best_decode_${decode_format}_denorm
asr_name=librispeech.transformer.ngpu4_asr.result.v2
asv_name=xvector_nnet_1a_asv.result

for tgtspk in TEF1 TEF2 TEM1 TEM2; do
    pair_name=Any_${tgtspk}_test
    MCD=$(cat ${expdir}/Any_${tgtspk}_test/mcd.log | tail -n 3 | head -n 1)
    CER=$(cat ${expdir}.ob_eval/${asr_name}/${pair_name}/result.txt | head -n 11 | tail -n 1 | awk '{print $11}')
    WER=$(cat ${expdir}.ob_eval/${asr_name}/${pair_name}/result.wrd.txt | head -n 11 | tail -n 1 | awk '{print $11}')
    PLDA=$(cat ${expdir}.ob_eval/${asv_name}/${tgtspk}/plda_scores | tail -n 1 | awk '{print $3}')
    COS=$(cat ${expdir}.ob_eval/${asv_name}/${tgtspk}/cossim_scores | tail -n 1 | awk '{print $3}')
    echo "${tgtspk}-${decode_format}  MCD: ${MCD}  CER: ${CER}  WER: ${WER}  PLDA: ${PLDA}  COSSIM: ${COS}"
done


