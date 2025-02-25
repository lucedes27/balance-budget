from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import ModelLoadConfig, LoraConfig, SFTRunConfig, TrainingArgumentsConfig, sft_batch_size, effective_batch_size

from tuning.utils.utils import apply_chat_template, chat_template_func
import json
import sys

def train_model_sft(
    run_config: SFTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: TrainingArgumentsConfig = None,
):  

    train_batch_size = sft_batch_size(run_config.dataset_config.train_size)

    gradient_accumulation_steps = effective_batch_size(run_config.dataset_config.train_size) // train_batch_size

    dataset = get_train_dataset(run_config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = run_config.model_name_hf,
        max_seq_length = model_load_config.max_seq_length,
        dtype = model_load_config.dtype,
        load_in_4bit = model_load_config.load_in_4bit,
    )

    dataset = apply_chat_template(tokenizer, dataset)
    
    print(dataset)  
    print(dataset["train"][0])

    if run_config.model_name == "qwen2-7B":
        lora_config.target_modules.extend(["embed_tokens", "lm_head"])

    
    model = FastLanguageModel.get_peft_model(
        model,  
        r = lora_config.r,
        target_modules = lora_config.target_modules,
        lora_alpha = lora_config.lora_alpha, 
        lora_dropout = lora_config.lora_dropout,
        bias = lora_config.bias,
        use_gradient_checkpointing = lora_config.use_gradient_checkpointing,
        random_state = lora_config.random_state, 
        use_rslora = lora_config.use_rslora,
        loftq_config = lora_config.loftq_config,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = chat_template_func(tokenizer),
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = model_load_config.max_seq_length,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            per_device_eval_batch_size = training_args.per_device_eval_batch_size,
            eval_steps = training_args.eval_steps,
            do_eval = training_args.do_eval,
            eval_strategy = training_args.eval_strategy,
            warmup_ratio = training_args.warmup_ratio,
            num_train_epochs = training_args.num_train_epochs,
            learning_rate = training_args.learning_rate,
            optim = training_args.optim,
            weight_decay = training_args.weight_decay,
            lr_scheduler_type = training_args.lr_scheduler_type,
            report_to = training_args.report_to,
            logging_steps = training_args.logging_steps,
            output_dir = run_config.output_dir,
            save_strategy="no",
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            seed = 42,
        ),
    )
    args = trainer.args.to_dict()

    print(args)
    trainer_stats = trainer.train()

    model.save_pretrained_merged(run_config.output_dir, tokenizer, save_method = "merged_16bit")

    with open(f"{run_config.output_dir}/training_config.json", "w") as f:
        json.dump(args, f, indent=4)



if __name__ == "__main__":
    from tuning.training.config_training import DatasetConfig, SFTRunConfig
    from tuning.config import MODELS_DIR

    dataset_config = DatasetConfig(
        dataset = "combined",
        dataset_type = "sft",
        train_size = 5000,
    )

    print(dataset_config)

    run_config = SFTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = "unsloth/Qwen2.5-7B",
        model_name = f"{MODELS_DIR}/llama3-8B_pt-combined-5000",
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )

    print(run_config)

    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = TrainingArgumentsConfig()


    train_model_sft(
        run_config = run_config,
        lora_config = lora_config,
        model_load_config = model_load_config,
        training_args = training_args,
    )



