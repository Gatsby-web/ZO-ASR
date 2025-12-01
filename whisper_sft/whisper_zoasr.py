import argparse
from datasets import load_dataset, concatenate_datasets, Audio
import datasets, torch
import random
import wandb
import os
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

from utils.collators import DataCollatorSpeechSeq2SeqWithPadding
from utils.dataset_preparation import prepare_dataset
from utils.trainer_zoasr import OurTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Whisper model for ASR")
    parser.add_argument("--asr_task", type=str, default="train", choices=["train","eval"], help="choose asr task")
    parser.add_argument("--whisper_path", type=str, default="/path/to/whisper-large-v3", help="Path to the Whisper model")
    parser.add_argument("--data_path", type=str, default="/path/to/common_voice_11_0", help="Path to dataset")
    parser.add_argument("--language", type=str, default="Hindi", help="Language for Whisper model")
    parser.add_argument("--task", type=str, default="transcribe", help="Task for Whisper model")
    parser.add_argument("--output_dir", type=str, default="./checkpoint_zoasr", help="Path for output checkpoint")
    parser.add_argument("--train_batch_size", type=int, default=24, help="Per device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=24, help="Per device evaluate batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=30000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save model every n steps")
    parser.add_argument("--eval_steps", type=int, default=2000, help="Evaluate model every n steps")
    parser.add_argument("--checkpoint_dir", type=str, default="/path/to/save/whisper-large-v3", help="Path to saved checkpoints")
    parser.add_argument("--metrics", type=str, default="wer", choices=["wer", "cer"], help="Choose evaluation metric: 'wer' or 'cer'")

    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"], help="Optimizer to use, either 'sgd' or 'adamw'")
    parser.add_argument("--momentum", type=float, default=0, help="Momentum for SGD (only applicable if optimizer is 'sgd')")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for optimizer")

    parser.add_argument("--trainer", type=str, default="zo", help="trainer")
    parser.add_argument("--zo_eps", type=float, default=1e-3, help="epsilon")
    parser.add_argument("--seed", type=int, default=6, help="random seed")
    parser.add_argument("--project", type=str, default="whisper-large-v3-zoasr-q8", help="project name for wandb")
    parser.add_argument("--q", type=int, default=8, help="query number")

    parser.add_argument("--start_layer", type=int, default=0, help="the first layer to optimize")
    parser.add_argument("--end_layer", type=int, default=-1, help="the last layer to optimize")


    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["WANDB_PROJECT"] = args.project
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = f"log/{args.project}.log"

    # load dataset
    common_voice = datasets.load_from_disk(args.data_path)
    common_voice["train"] = concatenate_datasets([common_voice["train"], common_voice["validation"]])
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    del common_voice["validation"]  
    del common_voice["other"]
    del common_voice["invalidated"] 
    #common_voice["test"] = common_voice["test"].select(random.sample(range(len(common_voice["test"])), 16))  # debug with 16 samples

    # load feature_extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.whisper_path, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.whisper_path, language=args.language, task=args.task)

    # resampling
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    common_voice = common_voice.map(lambda batch: prepare_dataset(batch, feature_extractor, tokenizer), 
                                    remove_columns=common_voice.column_names["train"], num_proc=1)

    # load checkpoint
    if args.asr_task == "train":
        #model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint_dir)
        model = WhisperForConditionalGeneration.from_pretrained(args.whisper_path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint_dir)
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    if args.metrics == "wer":
        metric = evaluate.load("utils/wer.py")
    elif args.metrics == "cer":
        metric = evaluate.load("utils/cer.py")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        score = metric.compute(predictions=pred_str, references=label_str)
        return {args.metrics: score}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=False,
        fp16=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=5,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model=args.metrics,
        greater_is_better=False,
        push_to_hub=False,
        lr_scheduler_type="constant",
    )

    # optimer, please use sgd optimizer if q>1
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )

    trainer_args = {
        "custom_trainer": args.trainer,
        "custom_eps": args.zo_eps,
        "log_path": log_filename,
        "q": args.q,
        "seed": args.seed,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer
    }

    trainer = OurTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        **trainer_args
    )
    trainer.optimizer = optimizer

    if args.asr_task == "train":
        #trainer.train(resume_from_checkpoint=True)
        trainer.train()
    elif args.asr_task == "eval":
        evaluation_results = trainer.evaluate()
        print(evaluation_results)  

if __name__ == "__main__":
    main()
