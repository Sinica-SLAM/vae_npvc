#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*" >&2 # Print the command line for logging
. ./path.sh

nj=1
cmd=run.pl
nlsyms=""
lang=""
feat_in=""  # feat.scp
feat_out="" # feat.scp
oov="<unk>"
bpecode=""
allow_one_column=false
verbose=0
trans_type=char
filetype=""
preprocess_conf=""
category=""
out="" # If omitted, write in stdout

text=""
multilingual=false

help_message=$(cat << EOF
Usage: $0 <data-dir>
e.g. $0 data/train
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --feat-in <feat-scp>                             # feat.scp or feat1.scp,feat2.scp,...
  --feat-out <feat-scp>                            # feat.scp or feat1.scp,feat2.scp,...
  --oov <oov-word>                                 # Default: <unk>
  --out <outputfile>                               # If omitted, write in stdout
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
  --preprocess-conf <json>                         # Apply preprocess to feats when creating shape.scp
  --verbose <num>                                  # Default: 0
EOF
)
. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

dir=$1
tmpdir=$(mktemp -d ${dir}/tmp-XXXXX)
trap 'rm -rf ${tmpdir}' EXIT

# 1. Create scp files for inputs
#   These are not necessary for decoding mode, and make it as an option
input=
if [ -n "${feat_in}" ]; then
    _feat_scps=$(echo "${feat_in}" | tr ',' ' ' )
    read -r -a feat_scps <<< $_feat_scps
    num_feats=${#feat_scps[@]}

    for (( i=1; i<=num_feats; i++ )); do
        feat=${feat_scps[$((i-1))]}
        mkdir -p ${tmpdir}/input_${i}
        input+="input_${i} "
        cat ${feat} > ${tmpdir}/input_${i}/feat.scp

        # Dump in the "legacy" style JSON format
        if [ -n "${filetype}" ]; then
            awk -v filetype=${filetype} '{print $1 " " filetype}' ${feat} \
                > ${tmpdir}/input_${i}/filetype.scp
        fi

        feat_to_shape.sh --cmd "${cmd}" --nj ${nj} \
            --filetype "${filetype}" \
            --preprocess-conf "${preprocess_conf}" \
            --verbose ${verbose} ${feat} ${tmpdir}/input_${i}/shape.scp
    done
fi

# 2. Create scp files for outputs
output=
if [ -n "${feat_out}" ]; then
    _feat_scps=$(echo "${feat_out}" | tr ',' ' ' )
    read -r -a feat_scps <<< $_feat_scps
    num_feats=${#feat_scps[@]}

    for (( i=1; i<=num_feats; i++ )); do
        feat=${feat_scps[$((i-1))]}
        mkdir -p ${tmpdir}/output_${i}
        output+="output_${i} "
        cat ${feat} > ${tmpdir}/output_${i}/feat.scp

        # Dump in the "legacy" style JSON format
        if [ -n "${filetype}" ]; then
            awk -v filetype=${filetype} '{print $1 " " filetype}' ${feat} \
                > ${tmpdir}/output_${i}/filetype.scp
        fi

        feat_to_shape.sh --cmd "${cmd}" --nj ${nj} \
            --filetype "${filetype}" \
            --preprocess-conf "${preprocess_conf}" \
            --verbose ${verbose} ${feat} ${tmpdir}/output_${i}/shape.scp
    done
fi

# 3. Create scp files for the others
mkdir -p ${tmpdir}/other
if [ ${multilingual} == true ]; then
    awk '{
        n = split($1,S,"[-]");
        lang=S[n];
        print $1 " " lang
    }' ${text} > ${tmpdir}/other/lang.scp
elif [ -n "${lang}" ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${text} > ${tmpdir}/other/lang.scp
fi

if [ -n "${category}" ]; then
    awk -v category=${category} '{print $1 " " category}' ${dir}/text \
        > ${tmpdir}/other/category.scp
fi
cat ${dir}/utt2spk > ${tmpdir}/other/utt2spk.scp

# 4. Merge scp files into a JSON file
opts=""
if [ -n "${feat_in}" -a -n "${feat_out}" ]; then
    intypes="${input} ${output} other"
else
    intypes="other"
fi
for intype in ${intypes}; do
    if [ -z "$(find "${tmpdir}/${intype}" -name "*.scp")" ]; then
        continue
    fi

    if [ ${intype} != other ]; then
        opts+="--${intype%_*}-scps "
    else
        opts+="--scps "
    fi

    for x in "${tmpdir}/${intype}"/*.scp; do
        k=$(basename ${x} .scp)
        if [ ${k} = shape ]; then
            opts+="shape:${x}:shape "
        else
            opts+="${k}:${x} "
        fi
    done
done

if ${allow_one_column}; then
    opts+="--allow-one-column true "
else
    opts+="--allow-one-column false "
fi

if [ -n "${out}" ]; then
    opts+="-O ${out}"
fi
merge_scp2json.py --verbose ${verbose} ${opts}

rm -fr ${tmpdir}
