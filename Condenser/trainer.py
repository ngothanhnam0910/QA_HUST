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

import os
from contextlib import nullcontext

from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers.trainer import Trainer
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
import datasets
try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False

import logging

logger = logging.getLogger(__name__)

class Trainer_v1(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        print("\n******Using Trainer_v1******\n")
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            #train_dataset = self._remove_unused_columns(train_dataset, description="training")
            description="training"
            if not self.args.remove_unused_columns:
                print("_remove_unused_columns - does not remove unsed columns")
            else:
                self._set_signature_columns_if_needed()
                signature_columns = self._signature_columns

                ignored_columns = list(set(train_dataset.column_names) - set(signature_columns))
                if len(ignored_columns) > 0:
                    dset_description = "" if description is None else f"in the {description} set"
                    logger.info(
                        f"The following columns {dset_description} don't have a corresponding argument in "
                        f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                        f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                        " you can safely ignore this message."
                    )

                columns = [k for k in signature_columns if k in train_dataset.column_names]

                if version.parse(datasets.__version__) < version.parse("1.4.0"):
                    train_dataset.set_format(
                        type=train_dataset.format["type"], columns=columns, format_kwargs=train_dataset.format["format_kwargs"]
                    )

                else:
                    train_dataset =  train_dataset.remove_columns(ignored_columns)
        else:
            print("remove columns collator")
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

            return dataloader

        train_sampler = self._get_train_sampler()
        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return dataloader

class CondenserPreTrainer(Trainer_v1):
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        # we are not going to do this in this
        # as collator will be generating new columns
        pass

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio

        super().create_optimizer_and_scheduler(num_training_steps)

    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')
        return model(inputs, labels)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop('labels')

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(inputs, labels)
            else:
                outputs = model(inputs, labels)

            loss = outputs

        return (loss, None, None)


class CoCondenserPretrainer(CondenserPreTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        super(CondenserPreTrainer, self).__init__(*args, **kwargs)

        if self.args.cache_chunk_size != -1:
            if not _grad_cache_available:
                raise ValueError(
                    'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
            self.gc = GradCache(
                models=[self.model.lm],
                chunk_sizes=self.args.cache_chunk_size,
                loss_fn=self.model.compute_contrastive_loss,
                get_rep_fn=lambda x: x.hidden_states[-1][:, 0],
                fp16=self.args.fp16,
                scaler=self.scaler
            )

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def compute_loss(self, model, inputs, grad_cache=None, chunk_offset=None):
        labels = inputs.pop('labels')
        return model(inputs, labels, grad_cache=grad_cache, chunk_offset=chunk_offset)

    def split_tensor_dict(self, td: Dict[str, Tensor]):
        keys = list(td.keys())
        chunked_tensors = [td[k].split(self.args.cache_chunk_size) for k in keys]
        return [dict(zip(keys, tt)) for tt in zip(*chunked_tensors)]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.args.cache_chunk_size == -1:
            return super(CoCondenserPretrainer, self).training_step(model, inputs)

        model.train()

        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop('labels')

        # Construct the gradient cache
        chunked_inputs = self.split_tensor_dict(inputs)
        for c in chunked_inputs:
            c['output_hidden_states'] = True
        cls_hiddens, rnd_states = self.gc.forward_no_grad(self.model.lm, chunked_inputs)
        if self.args.local_rank > -1:
            cls_hiddens = self.gather_tensors(cls_hiddens.contiguous())[0]
        grad_cache, total_loss = self.gc.build_cache(cls_hiddens)
        grad_cache = grad_cache[0]
        if self.args.local_rank > -1:
            total_loss = total_loss / dist.get_world_size()

        inputs['labels'] = labels
        chunked_inputs = self.split_tensor_dict(inputs)

        # Compute the full loss with cached gradients
        for local_chunk_id, chunk in enumerate(chunked_inputs):
            device_offset = max(0, self.args.local_rank) * self.args.per_device_train_batch_size * 2
            local_offset = local_chunk_id * self.args.cache_chunk_size
            chunk_offset = device_offset + local_offset
            with rnd_states[local_chunk_id]:
                if self.use_cuda_amp:
                    with autocast():
                        lm_loss, surrogate = self.compute_loss(model, chunk, grad_cache, chunk_offset)
                else:
                    lm_loss, surrogate = self.compute_loss(model, chunk, grad_cache, chunk_offset)

            if self.args.gradient_accumulation_steps > 1:
                raise ValueError

            ddp_no_sync = self.args.local_rank > -1 and (local_chunk_id + 1 < len(chunked_inputs))
            with model.no_sync() if ddp_no_sync else nullcontext():
                if self.use_cuda_amp:
                    (self.scaler.scale(lm_loss) + surrogate).backward()
                elif self.use_apex:
                    raise ValueError
                elif self.deepspeed:
                    raise ValueError
                else:
                    (lm_loss + surrogate).backward()
            total_loss += lm_loss
        return total_loss.detach()
