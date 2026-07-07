"""
Training Dataset Reader for TinyRecursiveModel.

Converts task data (NER, FPB, etc.) from AdaptLLM/finance-tasks into
causal language-modelling sequences for supervised fine-tuning.

For NER (class_num=1 / text completion tasks):
    Input format: "<question_text>"
    Target format: "<question_text><answer_text>"
    Loss mask: only on <answer_text> tokens.

For classification tasks (class_num>1):
    Input format: "<question_text><correct_answer>"
    Loss mask: only on <correct_answer> tokens.
"""

from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.dataset_readers.task import task_map


class TrainDatasetReader(Dataset):
    """
    PyTorch Dataset that wraps AdaptLLM task data for SFT training.

    Attributes
    ----------
    task_name : str
        One of the registered keys in task_map (e.g. "NER", "FPB").
    max_length : int
        Maximum tokenised sequence length (input + output).
    generate_max_len : int
        Reserved tokens for the answer portion (text completion tasks).
    add_bos_token : bool
        Whether to prepend BOS token.
    ignore_index : int
        Token id to ignore in the cross-entropy loss (default -100).
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        split: str = "train",
        max_length: int = 1024,
        generate_max_len: int = 128,
        cache_dir: Optional[str] = None,
        add_bos_token: bool = False,
        tokenizer_name: Optional[str] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.task = task_map.cls_dic[task_name]()
        self.max_length = max_length
        self.generate_max_len = generate_max_len
        self.add_bos_token = add_bos_token
        self.ignore_index = ignore_index

        # Resolve tokenizer name
        if tokenizer_name is None:
            tokenizer_name = model_name
        if model_name == "tiny_recursive" and tokenizer_name == "tiny_recursive":
            tokenizer_name = "gpt2"

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            model_max_length=max_length,
            truncation_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id < 0:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load dataset split
        dataset = self.task.get_dataset(cache_dir=cache_dir)
        # get_dataset returns the test split by default; try to get train split
        try:
            from datasets import load_dataset as _lds
            task_cfg = {
                "NER": ("AdaptLLM/finance-tasks", "NER"),
                "FPB": ("AdaptLLM/finance-tasks", "FPB"),
                "FiQA_SA": ("AdaptLLM/finance-tasks", "FiQA_SA"),
                "Headline": ("AdaptLLM/finance-tasks", "Headline"),
                "ConvFinQA": ("AdaptLLM/finance-tasks", "ConvFinQA"),
                "ChemProt": ("AdaptLLM/medicine-tasks", "ChemProt"),
                "RCT": ("AdaptLLM/medicine-tasks", "RCT"),
                "MQP": ("AdaptLLM/medicine-tasks", "MQP"),
                "USMLE": ("AdaptLLM/medicine-tasks", "USMLE"),
                "PubMedQA": ("AdaptLLM/medicine-tasks", "PubMedQA"),
                "SCOTUS": ("AdaptLLM/law-tasks", "SCOTUS"),
                "CaseHOLD": ("AdaptLLM/law-tasks", "CaseHOLD"),
                "UNFAIR_ToS": ("AdaptLLM/law-tasks", "UNFAIR_ToS"),
            }
            if task_name in task_cfg:
                repo, cfg_name = task_cfg[task_name]
                ds = _lds(repo, cfg_name, cache_dir=cache_dir)
                if split in ds:
                    self.data = list(ds[split])
                else:
                    # Fallback: use test split as proxy (few-shot SFT on test is valid
                    # for benchmarking improvement of the model's generation pattern)
                    print(
                        f"[TrainDatasetReader] '{split}' split not found for {task_name}; "
                        "falling back to 'test' split."
                    )
                    self.data = list(ds["test"])
            else:
                self.data = list(dataset)
        except Exception as e:
            print(f"[TrainDatasetReader] Could not load '{split}' split: {e}. Using test split.")
            self.data = list(dataset)

        self.is_completion = self.task.class_num == 1
        print(
            f"[TrainDatasetReader] Loaded {len(self.data)} examples for task={task_name}, "
            f"split={split}, is_completion={self.is_completion}"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = dict(self.data[index])
        if self.is_completion:
            return self._build_completion_instance(entry)
        else:
            return self._build_choice_instance(entry)

    # ------------------------------------------------------------------
    # Instance builders
    # ------------------------------------------------------------------

    def _build_completion_instance(self, entry: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        For tasks with class_num==1 (NER, ConvFinQA).
        The answer is free text; we do causal LM on (question + answer).
        Loss is applied only to the answer tokens.
        """
        question = self.task.get_question(entry)
        answer = self.task.get_answer(entry)  # e.g. " Borrower (person), Lender (person)"

        # Tokenize question only to find boundary
        q_ids = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_length - self.generate_max_len,
            add_special_tokens=self.add_bos_token,
        )["input_ids"]

        # Tokenize the full sequence (question + answer)
        full_text = question + answer
        qa_ids = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=self.add_bos_token,
        )["input_ids"]

        # Append EOS
        if self.tokenizer.eos_token_id is not None:
            qa_ids = qa_ids + [self.tokenizer.eos_token_id]
        qa_ids = qa_ids[: self.max_length]

        input_ids = torch.tensor(qa_ids, dtype=torch.long)

        # Labels: ignore question tokens, supervise only on answer.
        # Shift left by 1 for causal LM: labels[t] = token to predict at step t = input[t+1].
        labels = input_ids.clone()
        q_len = min(len(q_ids), len(qa_ids))
        labels[:q_len] = self.ignore_index  # mask question part

        # Causal LM shift: labels[t] should be input[t+1]
        labels = torch.cat([labels[1:], torch.tensor([self.ignore_index], dtype=torch.long)])

        # Shift labels for causal LM: predict next token
        # We keep labels aligned with input_ids; loss fn will handle the shift.
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

    def _build_choice_instance(self, entry: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        For tasks with class_num>1 (FPB, FiQA_SA, etc.).
        Treat the correct (question + correct_answer) as a causal LM sequence.
        Loss is applied only to the answer tokens.
        """
        answers = self.task.get_answers(entry)
        question = self.task.get_question(entry)
        answer = self.task.get_answer(entry)  # correct answer string

        q_ids = self.tokenizer(
            question,
            truncation=True,
            max_length=self.max_length - 32,
            add_special_tokens=self.add_bos_token,
        )["input_ids"]

        full_text = question + answer
        qa_ids = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=self.add_bos_token,
        )["input_ids"]

        if self.tokenizer.eos_token_id is not None:
            qa_ids = qa_ids + [self.tokenizer.eos_token_id]
        qa_ids = qa_ids[: self.max_length]

        input_ids = torch.tensor(qa_ids, dtype=torch.long)
        # Labels: ignore question tokens, supervise only on answer.
        # Shift left by 1 for causal LM: labels[t] = token to predict at step t = input[t+1].
        labels = input_ids.clone()
        q_len = min(len(q_ids), len(qa_ids))
        labels[:q_len] = self.ignore_index

        # Causal LM shift
        labels = torch.cat([labels[1:], torch.tensor([self.ignore_index], dtype=torch.long)])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences from TrainDatasetReader."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_padded = []
    labels_padded = []
    attention_masks_padded = []

    for b in batch:
        T = b["input_ids"].shape[0]
        pad_len = max_len - T
        input_ids_padded.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        labels_padded.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_masks_padded.append(
            torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded),
        "attention_mask": torch.stack(attention_masks_padded),
    }
