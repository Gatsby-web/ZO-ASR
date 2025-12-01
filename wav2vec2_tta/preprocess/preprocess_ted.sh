#! /bin/bash

audio_paths="/hpc_stor03/sjtu_home/yuezhang.peng/Test-time-adaptation-ASR-SUTA-main/TEDLIUM_release2/test/sph/*.sph"
output_dir="/hpc_stor03/sjtu_home/yuezhang.peng/Test-time-adaptation-ASR-SUTA-main/TEDLIUM_release2/test/wav"
[ ! -e "$output_dir" ] && mkdir "$output_dir"
for f in ${audio_paths}
do 
    IFS="/" read -ra arr <<< ${f}
    IFS="." read -ra name <<< ${arr[-1]}
    echo "filename: ${name}"
    sox $f "${output_dir}/${name}.wav"
done 
echo "done."