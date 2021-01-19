#!/bin/bash


if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriTTS/dev-clean data/dev-clean"
  exit 1
fi

src=$1
dst=$2

mkdir -p $dst
cp $src/{wav.scp,utt2spk,text.phone} $dst

trans=$dst/text
mv $dst/text.phone $trans

utt2spk=$dst/utt2spk
spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

ntrans=$(wc -l <$trans)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in $dst"

exit 0
