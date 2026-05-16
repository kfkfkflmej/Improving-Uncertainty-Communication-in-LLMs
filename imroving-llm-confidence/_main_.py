import os
import logging
import torch

from matplotlib import text
from omegaconf import OmegaConf
import logging
import hydra

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer



@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    dataset = load_dataset(dataset_filetype, data_files=dataset_filepath)
    model_id = model_base_model

    def formatting_prompts_func(example):
        return {
            "text": (
                f"User: {example['question']}\n"
                f"Assistant: {example['answer']}"
            )
    }
    processor = AutoProcessor.from_pretrained(model_id, token=os.environ['HF_TOKEN'])

    dataset = dataset.map(
    formatting_prompts_func,
    remove_columns=["question", "answer"]
    )

    logging.info(dataset)

    # Define model init arguments
    model_kwargs = dict(
        dtype=torch.bfloat16, # What torch dtype to use, defaults to auto
        device_map="auto", # Let torch decide how to load the model
    )

    # BitsAndBytesConfig int-4 config
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    # Load the Instruction Tokenizer to use the official Gemma template

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"], # make sure to save the lm_head and embed_tokens as you train the special tokens
        ensure_weight_tying=True,
    )

    args = SFTConfig(

        run_name=mapper_name,
        output_dir=sft_output_dir,     # directory to save and repository id
        num_train_epochs=sft_num_train_epochs,                    # number of training epochs
        per_device_train_batch_size=sft_per_device_train_batch_size,            # batch size per device during training
        gradient_accumulation_steps=sft_gradient_accumulation_steps,
        learning_rate=sft_learning_rate,
        bf16=sft_bf16,                                  # use bfloat16 precision
        dataset_text_field="",                      # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True}, # important for collator
        remove_unused_columns = False,             # important for collator
        max_length=sft_max_length,
        packing=sft_packing,
        warmup_steps=sft_warmup_steps,
        gradient_checkpointing=sft_gradient_checkpointing,
        fp16=sft_fp16,
        seed=sft_seed,
    )

    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]

        batch = processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

    # Create Trainer object
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        data_collator=collate_fn,
    )

    # Start training, the model will be SAVED TO THE OUTPUT DIRECTORY
    
    print_trainable_parameters(trainer.model)

    trainer.train()

    logging.info("Saving last checkpoint of the model")
    trainer.save_model(sft_args.output_dir)


if __name__ == "__main__":
    main()
