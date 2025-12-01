# ZO-ASR on Speech Foundation Model: Test Time Adaptation

This part of the code is for ZO-ASR experiments on Wav2Vec2-Base. It is based on [Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition](https://github.com/DanielLin94144/Test-time-adaptation-ASR-SUTA).

### Installation 
```
pip install -r requirements.txt
```
### Data Preparation
Please download datasets by your own: [Librispeech](https://www.openslr.org/12)/[CHiME-3](https://catalog.ldc.upenn.edu/LDC2017S24)/[Common voice En](https://tinyurl.com/cvjune2020)/[TED-LIUM 3](https://www.openslr.org/51/)

### Usage
The source ASR model is [w2v2-base fine-tuned on Librispeech 960 hours](https://huggingface.co/facebook/wav2vec2-base-960h). 

Run SUTA on different datasets:
```
bash scripts/LS_zoasr_sgd_40steps.sh
```

### Citation
```
@article{lin2022listen,
  title={Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition},
  author={Lin, Guan-Ting and Li, Shang-Wen and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2203.14222},
  year={2022}
}
```

