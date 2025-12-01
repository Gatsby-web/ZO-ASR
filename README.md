# ZO-ASR: Zeroth-Order Fine-Tuning of Speech Foundation Models without Back-Propagation

This repository contains the implementation of **ZO-ASR**. This project explores the application of zeroth-order optimization techniques on speech foundation models in two distinct scenarios: **Supervised Domain Adaptation** and **Test-Time Adaptation**.

## Repository Structure

### 1. Supervised Domain Adaptation (`whisper_sft/`)
- **Model:** Whisper-Large-V3
- **Goal:** Adapting the model to specific languages (e.g., Hindi) using limited GPU resources.
- **Base Code:** Adapted from [Hugging Face Whisper Fine-tuning](https://huggingface.co/blog/fine-tune-whisper) and [MeZO](https://github.com/princeton-nlp/MeZO).
- **Details:** Please refer to [whisper_sft/README.md](whisper_sft/README.md) for installation and training scripts.

### 2. Test-Time Adaptation (`wav2vec2_tta/`)
- **Model:** Wav2Vec2-Base
- **Goal:** Improving WER on unseen test data without access to the source training data.
- **Base Code:** Based on [SUTA](https://github.com/DanielLin94144/Test-time-adaptation-ASR-SUTA).
- **Details:** Please refer to [wav2vec2_tta/README.md](wav2vec2_tta/README.md) for data preparation and evaluation scripts.

## Getting Started

Since the two experiments rely on different base libraries and dependencies, please navigate to the respective directories to set up the environment:

**For Whisper Domain Adaptation:**
```bash
cd whisper_sft
# Follow the instructions in whisper_sft/README.md
````

**For Wav2Vec2 Test-Time Adaptation:**

```bash
cd wav2vec2_tta
# Follow the instructions in wav2vec2_tta/README.md
```

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{peng2025,
  title={ZO-ASR: Zeroth-Order Fine-Tuning of Speech Foundation Models without Back-Propagation},
  author={Yuezhang Peng and Yuxin Liu and Yao Li and Sheng Wang and Fei Wen and Xie Chen},
  journal={2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year={2025}
}

