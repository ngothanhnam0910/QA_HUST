# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
# Copyright 2021 Condenser Author All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import sys
from datasets import load_dataset

from arguments import DataTrainingArguments, ModelArguments, \
    CoCondenserPreTrainingArguments as TrainingArguments
from data import CoCondenserDataset, CoCondenserCollator
from modeling import CoCondenserForPretraining
from trainer_old import CoCondenserPretrainer as Trainer
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed, )
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

CONDENSER_TYPE_MAP = {
    'bert': CoCondenserForPretraining,
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    # training_args = TrainingArguments(output_dir='/home/gemai/md1/NAMNT_DA2/checkpoint/ckp_pretrain/ckp_cocondenser/', overwrite_output_dir=True,
    # do_train=True, do_eval=True, do_predict=False, evaluation_strategy = 'steps',
    # prediction_loss_only=False, per_device_train_batch_size=16, per_device_eval_batch_size=8, per_gpu_train_batch_size=None,
    # per_gpu_eval_batch_size=None, gradient_accumulation_steps=1, eval_accumulation_steps=None, learning_rate=1e-05, weight_decay=0.01, adam_beta1=0.9,
    # adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=50.0, max_steps=-1, lr_scheduler_type='linear',
    # warmup_steps=0, logging_dir='runs/Apr17_17-20-46_gemai', logging_first_step=False, logging_steps=500, save_steps=100, save_total_limit=None, no_cuda=False, seed=42, fp16=True, fp16_opt_level='O1',
    # fp16_backend='auto', local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False,
    # dataloader_drop_last=True, eval_steps=500, dataloader_num_workers=8, past_index=-1,
    # run_name='/home/gemai/md1/NAMNT_DA2/checkpoint/ckp_pretrain/ckp_cocondenser/', disable_tqdm=False,
    # remove_unused_columns= False, label_names=None, load_best_model_at_end=False, metric_for_best_model=None,
    # greater_is_better=None, ignore_data_skip=False, sharded_ddp=False, deepspeed=None, label_smoothing_factor=0.0,
    # adafactor=False, warmup_ratio=0.1, cache_chunk_size=32)

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_set = CoCondenserDataset(load_dataset(
        'json',
        data_files=data_args.train_path,
        block_size=2 ** 25,
        ignore_verifications=False,
    )['train'], data_args)
    dev_set = None

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir, use_fast=False
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=False
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # initialize the Condenser Pre-training LMX
    if model_args.model_type not in CONDENSER_TYPE_MAP:
        raise NotImplementedError(f'Condenser for {model_args.model_type} LM is not implemented')
    _condenser_cls = CONDENSER_TYPE_MAP[model_args.model_type]
    if model_args.model_name_or_path:
        model = _condenser_cls.from_pretrained(
            model_args, data_args, training_args,
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.warning('Training from scratch.')
        model = _condenser_cls.from_config(
            config, model_args, data_args, training_args)

    model.lm.resize_token_embeddings(len(tokenizer))

    # Data collator
    data_collator = CoCondenserCollator(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        max_seq_length=data_args.max_seq_length,
    )
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        # trainer.train(model_path=model_path)
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
