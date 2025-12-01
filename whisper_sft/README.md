# ZO-ASR on Speech Foundation Model: Supervised Domain Adaptation

This part of the code is for ZO-ASR experiments on Whisper-Large-V3. It is based on [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper) and [MeZO](https://github.com/princeton-nlp/MeZO).

## Installation

Please install the versions of PyTorch (`pytorch` following [https://pytorch.org](https://pytorch.org)) and Transformers (`transformers`). This code is tested on `torch==2.1.0.dev20230514+cu118` and `transformers==4.45.1` with Python 3.10.15.

## Prepare the data

Datasets link is [here](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0). Please download it and extract the files to `./common_voice`, or run the following commands:

```bash
python download_dataset.py
```
You can change the path or language settings in the file as needed.

Note that we follow the setting in [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper) by combining the training and validation sets to form a larger training set, which is especially useful given the limited resources available for languages such as Hindi. 


## Usage

Use `whisper_zoasr.py` for all functions and arguments. The GPU memory usage for Whisper-Large-V3 with batch_size=24 and SGD optimizer is less than 24G and can be run on a single RTX 3090.

Here is an example:

```bash
export CUDA_VISIBLE_DEVICES=0
python whisper_zoasr.py \
    --asr_task train \
    --whisper_path /path/to/whisper-large-v3 \
    --data_path /path/to/dataset \
    --language hi \
    --seed 6 \
    --output_dir ./output_zoasr/output_zoasr_whisper-large-v3-hi-q8 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --learning_rate 3e-6 \
    --max_steps 30000 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --optimizer sgd \
    --zo_eps 1e-3 \
    --q 8 \
    --project whisper-large-v3-nspsa-pt
```


