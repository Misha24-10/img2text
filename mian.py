import os

from src.dataset import CocoDataset, CustomCollator
from src.trainer import CustomTrainer, training_args
from src.model import model, text_tokenizer, feature_extractor
from src.config import final_save, coco_dir, path_annot_train, path_annot_val

if __name__ == "__main__":
    ds_train = CocoDataset(coco_dir=coco_dir, annotation_file=path_annot_train, split="val")
    ds_val = CocoDataset(coco_dir=coco_dir, annotation_file=path_annot_val, split="val")

    collator = CustomCollator()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        compute_metrics=None
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(final_save)
    text_tokenizer.save_pretrained(final_save)
    feature_extractor.save_pretrained(final_save)
        