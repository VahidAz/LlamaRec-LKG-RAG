import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
from datasets import DATASETS, dataset_factory
from config import *
from model import *
from dataloader import *
from trainer import *

from transformers import BitsAndBytesConfig
from pytorch_lightning import seed_everything
from model import LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    PeftConfig,
    PeftModel
)


try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')
os.environ["WANDB_DISABLED"] = "true"


def main(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = LlamaForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,
        device_map='auto',
        cache_dir=args.llm_cache_dir,
    )

    # Adding user preferences module
    if args.llm_train_with_relation_score:
        dataset = dataset_factory(args)
        path = str(dataset._get_preprocessed_folder_path()) + "/dataset.pkl"
        with open(path, "rb") as fin:
            preprocess_dataset = pickle.load(fin)
        model.add_user_preferences_module(len(preprocess_dataset["umap"]) + 1, 
                                          args.llm_user_embedding_dim, 
                                          len(args.llm_relations_class))

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if args.llm_only_eval:
        if not args.llm_checkpoint_path:
            raise ValueError("The path for the checkpoint is not provided.")
        config = PeftConfig.from_pretrained(args.llm_checkpoint_path)
        model = PeftModel.from_pretrained(model, args.llm_checkpoint_path)
        model.load_adapter(args.llm_checkpoint_path, adapter_name='adp1')
        model.set_adapter("adp1")
        model.print_trainable_parameters()
    elif args.llm_continue_training:
        model = PeftModel.from_pretrained(model, args.llm_checkpoint_path, is_trainable=True)
    else:
        if args.llm_train_with_relation_score:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias='none',
                task_type="CAUSAL_LM",
                modules_to_save=["user_preferences.embedding",
                                 "user_preferences.layer_norm",
                                 "user_preferences.embed_dropout",
                                 "user_preferences.fc1"],
            )
        else:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias='none',
                task_type="CAUSAL_LM",
            )
        
        if not args.llm_continue_training:
            model = get_peft_model(model, config)
        model.print_trainable_parameters()
    
    train_loader, val_loader, test_loader, tokenizer, test_retrieval = dataloader_factory(args, model)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Data type: {param.dtype}")

    model.config.use_cache = False
    trainer = LLMTrainer(args, model, train_loader, val_loader, test_loader, tokenizer, export_root, args.use_wandb)
    
    if not args.llm_only_eval:
        trainer.train()
    trainer.test(test_retrieval)


if __name__ == "__main__":
    args.model_code = 'llm'
    args.dataset_code = args.llm_retrieved_path.split("/")[-1]
    set_template(args)
    if args.llm_train_with_relation:
        args.llm_max_text_len = args.llm_max_text_len_relation
    if args.llm_train_with_relation_score and not args.llm_train_with_relation:
        raise ValueError("llm_train_with_relation must be set to True if you mean training with user preferences.")
    main(args, export_root=None)