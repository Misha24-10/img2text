import numpy as np
import random
import torch

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch import nn
from pprint import pprint
from torchmetrics.text import BLEUScore, ROUGEScore
from src.model import text_tokenizer
from src.config import Trainerconfig

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.eval_compute_metrics

    def eval_compute_metrics(self, eval_preds):
        preds, labels, metadata = eval_preds
        ignore_pad_token_for_loss = True
        preds, labels, _ = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = text_tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, text_tokenizer.pad_token_id)

        decoded_labels = self.atguments_for_evaluate
        indexes_to_print = random.sample(range(len(decoded_labels)), k=5)
        print("Sample Predictions:")
        pprint([decoded_preds[index] for index in indexes_to_print])
        print("Sample GT labels:")
        pprint([decoded_labels[index] for index in indexes_to_print])

        result = {}
        prediction_lens = [np.count_nonzero(pred != text_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        bleu_scores = {f"BLEU_{i}": BLEUScore(n_gram=i) for i in range(1, 5)}
        for key, bleu in bleu_scores.items():
            result[key] = float(bleu(decoded_preds, decoded_labels))
            print(f"{key}: {result[key]}")

        rouge = ROUGEScore()
        rouge_result = rouge(decoded_preds, decoded_labels)

        for key, value in rouge_result.items():
            result[key] = value.item()
            print(f"{key}: {result[key]}")

        self.atguments_for_evaluate = []
        return result

    def training_step(self, model: nn.Module, inputs: dict) -> torch.Tensor:
        inputs.pop('gt_annotations', None)
        model.train()
        inputs = self._prepare_inputs(inputs)
        outputs = super().training_step(model, inputs)
        return outputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not hasattr(self, 'atguments_for_evaluate'):
            self.atguments_for_evaluate = []
        gt_annotations = inputs.pop('gt_annotations', None)
        self.atguments_for_evaluate.extend(gt_annotations)
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        inputs["gt_annotations"] = gt_annotations
        return outputs

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=Trainerconfig.predict_with_generate,
    evaluation_strategy=Trainerconfig.evaluation_strategy,
    logging_steps=Trainerconfig.logging_steps,
    num_train_epochs=Trainerconfig.num_train_epochs,
    warmup_steps=Trainerconfig.warmup_steps,
    eval_steps=Trainerconfig.eval_steps,
    save_steps=Trainerconfig.save_steps,
    learning_rate=Trainerconfig.learning_rate,
    save_total_limit=Trainerconfig.save_total_limit,
    per_device_train_batch_size=Trainerconfig.per_device_train_batch_size,
    per_device_eval_batch_size=Trainerconfig.per_device_eval_batch_size,
    output_dir=Trainerconfig.output_dir,
    # fp16=Trainerconfig.fp16,
    dataloader_num_workers=Trainerconfig.dataloader_num_workers,
    remove_unused_columns=Trainerconfig.remove_unused_columns,
    run_name=Trainerconfig.run_name,
    include_inputs_for_metrics=Trainerconfig.include_inputs_for_metrics,
    report_to="wandb"
)