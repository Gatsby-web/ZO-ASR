from datasets import load_dataset, DatasetDict
common_voice = DatasetDict()
common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "hi",cache_dir="./common_voice_11_0_hi")
common_voice.save_to_disk('common_voice_11_0_hi')
print(common_voice)