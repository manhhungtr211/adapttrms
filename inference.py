import warnings
import torch
import tqdm
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import hydra.utils as hu
import hydra
import json
import os
import shutil
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader
from accelerate import Accelerator

from src.utils.metric import compute_scores
import glob
import logging
from transformers import StoppingCriteriaList, StoppingCriteria



logger = logging.getLogger(__name__)

class StopStrCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_str, answer_start):
        self.stop_str = stop_str
        self.tokenizer = tokenizer
        self.answer_start = answer_start

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        sequences = self.tokenizer.batch_decode(input_ids[:, self.answer_start:])
        return all(self.stop_str in seq for seq in sequences)

class Inferencer:
    def __init__(self, cfg, accelerator, model) -> None:
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.dataset_reader.shard(accelerator)
        if self.dataset_reader.tokenizer.pad_token_id is None or self.dataset_reader.tokenizer.pad_token_id <0:
            self.dataset_reader.tokenizer.pad_token_id = self.dataset_reader.tokenizer.eos_token_id

        self.accelerator = accelerator
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=accelerator.device)

        self.dataloader = DataLoader(self.dataset_reader, batch_size=self.dataset_reader.task.inf_bsz, collate_fn=co)
        self.model=model

        self.output_file = os.path.join(cfg.output_dir, f'{os.path.basename(cfg.model_name)}_{cfg.task_name}.json')
        self.res_file = os.path.join(cfg.res_dir, f'{cfg.task_name}.txt')
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.res_file), exist_ok=True)
        
        self.cfg = cfg
        self.option_num = self.dataset_reader.task.class_num
        self.max_length = cfg.max_length  # used for text completion task,
        self.generate_max_len = cfg.generate_max_len  # max seq len to be generated

    def choice_losses(self, input_ids, input_atten_mask, loss_mask, labels):
        bsz, option_num, seq_len = input_ids.shape
        if self.option_num is not None:
            assert option_num == self.option_num

        base_model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            output = base_model(
                input_ids=input_ids.reshape(bsz * option_num, seq_len),
                attention_mask=input_atten_mask.reshape(bsz * option_num, seq_len),
            )

        # (bsz, option_num, seq_len, vocab_size)
        logits = output.logits.reshape(bsz, option_num, seq_len, -1)

        # (bsz, option_num, seq_len-1, vocab_size)
        logits = logits[:, :, :-1, :]

        # (bsz, option_num, seq_len-1, 1)
        targets = input_ids[:, :, 1:].unsqueeze(-1)

        vocab_size = None
        if hasattr(base_model, 'vocab_size'):
            vocab_size = base_model.vocab_size
        elif hasattr(base_model, 'config') and hasattr(base_model.config, 'vocab_size'):
            vocab_size = base_model.config.vocab_size

        loss_mask = loss_mask[:, :, 1:]
        if vocab_size is not None:
            invalid_target = targets >= vocab_size
            if invalid_target.any():
                loss_mask = loss_mask.masked_fill(invalid_target.squeeze(-1), 0)
                targets = targets.clamp(max=vocab_size - 1)

        # (bsz, option_num, seq_len-1, vocab_size)
        logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

        # (bsz, option_num, seq_len-1,1), squeeze to (bsz, option_num, seq_len-1), loss mask to (bsz, option_num, answer_len)
        loss = - torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask

        safe_den = loss_mask.sum(-1).clamp(min=1)
        loss = loss.sum(-1) / safe_den

        # (bsz,)
        preds = torch.argmin(loss, dim=-1).detach().cpu().tolist()
        normed_loss = torch.nn.functional.normalize(loss, p=1, dim=-1)
        
        # roughly get pred_probs for all classes
        pred_probs = 2/option_num - normed_loss
        pred_probs = pred_probs.detach().cpu().tolist()
        labels = labels.squeeze(-1).tolist()
        assert len(labels) == len(preds)
        return {"preds": preds, "labels": labels, "pred_probs": pred_probs}

    def completion_losses(self, input_ids, input_atten_mask, labels):
        bsz, option_num, seq_len = input_ids.shape
        answer_start = int(input_atten_mask.shape[-1])
        stopping_criteria = StoppingCriteriaList([StopStrCriteria(tokenizer=self.dataset_reader.tokenizer, 
                                                                stop_str='\n', answer_start=answer_start)])
        assert option_num == 1
        base_model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            res = base_model.generate(
                input_ids=input_ids.squeeze(1),  # remove the dim of option_num
                attention_mask=input_atten_mask.squeeze(1),
                pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                max_length=min(self.max_length, answer_start + self.generate_max_len),
                do_sample=False,
                stopping_criteria=stopping_criteria,
            )
        pred_ids = res[:, answer_start:]
        preds = []
        for i in range(len(pred_ids)):
            preds.append(self.dataset_reader.tokenizer.decode(pred_ids[i], skip_special_tokens=True))
        return {"preds": preds, "labels": labels, "pred_probs": [None] * len(preds)}

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        cached_file_path = f"{self.output_file}tmp_{self.accelerator.device}.bin"
        with BufferedJsonWriter(cached_file_path) as buffer:
            if os.path.isfile(cached_file_path):
                os.remove(cached_file_path)
                print("cached file: % s removed successfully" % cached_file_path) 
            for i, entry in enumerate(dataloader):
                if "stop" in self.cfg and i == self.cfg.stop:
                    break  # early stop for debug
                metadata = entry.pop("metadata")
                
                # unwrap ListWrapper -> list
                if hasattr(metadata, "data"):
                    metadata = metadata.data
                else:
                    metadata = list(metadata)  # fallback nếu __iter__ có define

                # print(type(metadata), metadata)
                
                if self.dataset_reader.task.class_num == 1:
                    few_shot_res = self.completion_losses(
                        input_ids=entry.input_ids,
                        input_atten_mask=entry.input_atten_mask,
                        labels=[x["label"] for x in metadata],
                    )
                else:
                    few_shot_res = self.choice_losses(
                        input_ids=entry.input_ids,
                        input_atten_mask=entry.input_atten_mask,
                        loss_mask=entry.loss_mask,
                        labels=entry.labels,
                    )
                for i in range(len(metadata)):
                    metadata[i]["pred_prob"] = few_shot_res["pred_probs"][i]
                    metadata[i]["pred"] = few_shot_res["preds"][i]
                    metadata[i]["label"] = few_shot_res["labels"][i]
                buffer.write(metadata)

    def write_predictions(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with BufferedJsonReader(path) as f:
                for x in f.read():
                    data.extend(x)
        logger.info("num of saved preds: %s", str(len(data)))
        scores = compute_scores(self.dataset_reader.task.metric, data)

        with open(self.output_file, "w") as f:
            f.write(json.dumps(data, indent=4) + "\n")
        logger.info("scores: %s", str(scores))
        with open(self.res_file, "a") as f:
            info = f"model: {str(self.cfg.model_name)}; scores: {str(scores)}\n"
            f.write(info)
        
        print("saved pred to: ", self.output_file)
        print("saved eval res to: ", self.res_file)
        # Copy outputs to a local folder (use cfg.local_output_dir or Kaggle working dir if available)
        try:
            local_dir = getattr(self.cfg, 'local_output_dir', None)
            if local_dir is None:
                if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle/working'):
                    local_dir = '/kaggle/working'
                else:
                    local_dir = os.getcwd()
            os.makedirs(local_dir, exist_ok=True)
            dest_out = os.path.join(local_dir, os.path.basename(self.output_file))
            dest_res = os.path.join(local_dir, os.path.basename(self.res_file))
            shutil.copy(self.output_file, dest_out)
            shutil.copy(self.res_file, dest_res)
            logger.info("copied outputs to local: %s", local_dir)
            print("downloaded pred to:", dest_out)
            print("downloaded eval res to:", dest_res)
            # If the task is NER, print the contents of the downloaded files for diagnosis
            try:
                if getattr(self.cfg, 'task_name', '').upper() == 'NER':
                    print('---- START OF DOWNLOADED PRED FILE ----')
                    with open(dest_out, 'r', encoding='utf-8') as df:
                        for line in df:
                            print(line.rstrip())
                    print('---- END OF DOWNLOADED PRED FILE ----')

                    print('---- START OF DOWNLOADED RES FILE ----')
                    with open(dest_res, 'r', encoding='utf-8') as rf:
                        for line in rf:
                            print(line.rstrip())
                    print('---- END OF DOWNLOADED RES FILE ----')
            except Exception as e:
                logger.warning('failed to print downloaded files for diagnosis: %s', str(e))
            # If entity-level F1 exists and is zero, save a small sample for inspection
            try:
                if float(scores.get('entity_level_f1', 0.0)) == 0.0:
                    sample = data[:5]
                    sample_path = os.path.join(local_dir, f'sample_{os.path.basename(self.output_file)}')
                    with open(sample_path, 'w') as sf:
                        sf.write(json.dumps(sample, indent=2) + "\n")
                    logger.warning("entity_level_f1 is 0. Saved first 5 preds to %s for inspection", sample_path)
                    print("Saved sample preds to:", sample_path)
            except Exception:
                pass
        except Exception as e:
            logger.warning("failed to copy outputs locally: %s", str(e))
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)
        return data


@hydra.main(config_path="configs", config_name="inference")
def main(cfg):
    logger.info(cfg)

    accelerator = Accelerator()
    # load model once and run for many tasks
    model = hu.instantiate(cfg.model).half()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameter count: {total_params:,}')
    if cfg.model_type == "custom" or not cfg.model_parallel:
        model = model.to(accelerator.device)
        print('Mapped model to the local GPU for this process.')
    else:
        print('Using Accelerate distributed setup for multi-GPU execution.')
    model = accelerator.prepare(model)
    model = model.eval()

    # loop for tasks
    tasks = cfg.task_name.split('+')
    for task in tasks: 
        print(f'infer on {task}...')
        cfg.task_name = task
        inferencer = Inferencer(cfg, accelerator, model)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inferencer.forward()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                inferencer.write_predictions()


if __name__ == "__main__":
    main()
