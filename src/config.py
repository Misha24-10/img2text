import os
import wandb

use_wandb = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false" if use_wandb else "true"
os.environ["WANDB_PROJECT"] = "VIT_GPT"

if use_wandb:
    wandb.login()

image_encoder_model = "microsoft/beit-base-patch16-224-pt22k-ft22k"
text_decode_model = "GPT2"

output_dir = "./model_output_ENG"
final_save = "./final_model_beit_gpt"
coco_dir = "/home/misha/Desktop/education/2_semester/DL_course/nn_project/img2text/data/val2017/"
path_annot_train = "/home/misha/Desktop/education/2_semester/DL_course/nn_project/img2text/data/annotations_trainval2017/annotations/captions_val2017.json"
path_annot_val = "/home/misha/Desktop/education/2_semester/DL_course/nn_project/img2text/data/annotations_trainval2017/annotations/captions_val2017.json"


class ModelUpdate:
    max_length=5
    early_stopping=True
    no_repeat_ngram_size=3
    length_penalty=2
    num_beams=3

class Trainerconfig:
    predict_with_generate=True
    evaluation_strategy="steps"
    logging_steps=10
    num_train_epochs=9.0
    warmup_steps=1000
    eval_steps=5000
    save_steps=5000
    learning_rate=5e-5
    save_total_limit=2
    per_device_train_batch_size=2
    per_device_eval_batch_size=2
    output_dir=output_dir
    fp16=True
    dataloader_num_workers=2
    remove_unused_columns=False
    run_name="Model_Training_Run"
    include_inputs_for_metrics=True
