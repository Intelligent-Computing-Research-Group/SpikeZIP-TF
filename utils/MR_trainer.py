"""
Description: Run text classification on the given dataset with the given model.
"""
import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup, AdamW, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments, AutoTokenizer

from utils.data import Dataset, SimpleDataset, load_data, load_pretrained_embeddings, build_tokenizer_for_word_embeddings, \
    prepare_data_custom_tokenizer, prepare_data

try:
    import wandb

    WANDB = True
except ImportError:
    logging.info("Wandb not installed. Skipping tracking.")
    WANDB = False

MODELS = {
    "BERT": "bert-base-uncased",
    "ROBERTA": "/home/zkxu/SNN/code/roberta-base-mr",
    "DEBERTA": "microsoft/deberta-base",
    "MLP": "bert-base-uncased",
    "ERNIE": "nghuyong/ernie-2.0-base-en",
    "DISTILBERT": "distilbert-base-uncased",
    "ALBERT": "albert-base-v2",
    "LSTM": "bert-base-uncased",
}

def compute_metrics(pred):
    """
    Compute the metrics for the given predictions.
    :param pred: the predictions
    :return: accuracy
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def evaluate_trainer(trainer, test_data, output_dir):
    """
    Evaluate the fine-tuned trainer on the test set.
    Therefore, the accuracy is computed and a confusion matrix is generated.
    """

    # accuracy
    prediction_output = trainer.predict(test_data)
    logging.info(f"Prediction metrics: {prediction_output.metrics}")

    # confusion matrix
    y_preds = np.argmax(prediction_output.predictions, axis=1)
    y_true = prediction_output.label_ids
    cm = confusion_matrix(y_true, y_preds)
    logging.info(f"Confusion matrix:\n{cm}")

    # create file if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save results to file
    with open(f"{output_dir}/eval_results.json", "a") as f:
        f.write("\n")
        json.dump(prediction_output.metrics, f)

    if WANDB:
        wandb.log(prediction_output.metrics)

def train_transformer(model, dataset_name, output_dir, training_batch_size=128, eval_batch_size=128, learning_rate=4e-5,
                      num_train_epochs=30, weight_decay=0.0,
                      disable_tqdm=False):
    """
    Train and fine-tune the model using HuggingFace's PyTorch implementation and the Trainer API.
    """

    # training params
    model_ckpt = MODELS[model] if model in MODELS.keys() else model
    print(model_ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = f"{output_dir}/"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # max length of 512 for ERNIE as it is not predefined in the model
    logging.info(f"Loading {dataset_name} data...")
    dataset = load_data(dataset_name)
    if model == "ERNIE":
        test_data, train_data, label_dict = prepare_data(dataset, tokenizer, Dataset, max_length=512)
    else:
        test_data, train_data, label_dict = prepare_data(dataset, tokenizer, Dataset)

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=len(label_dict)).to(device)
    logging_steps = len(train_data) // training_batch_size

    # train
    if WANDB:
        wandb.watch(model)
        training_args = TrainingArguments(output_dir=output,
                                          num_train_epochs=num_train_epochs,
                                          learning_rate=learning_rate,
                                          per_device_train_batch_size=training_batch_size,
                                          per_device_eval_batch_size=eval_batch_size,
                                          weight_decay=weight_decay,
                                          evaluation_strategy="steps",  # 评估策略为steps
                                          eval_steps=50,  # 每500个steps进行一次评估
                                          disable_tqdm=disable_tqdm,
                                          logging_steps=logging_steps,
                                          log_level="error",
                                          logging_dir="./logs",
                                          load_best_model_at_end=True,  # 在训练结束时加载最好的模型
                                          metric_for_best_model="accuracy",
                                          report_to="wandb")
    else:
        training_args = TrainingArguments(output_dir=output,
                                          num_train_epochs=num_train_epochs,
                                          learning_rate=learning_rate,
                                          per_device_train_batch_size=training_batch_size,
                                          per_device_eval_batch_size=eval_batch_size,
                                          weight_decay=weight_decay,
                                          evaluation_strategy="steps",  # 评估策略为steps
                                          eval_steps=50,  # 每500个steps进行一次评估
                                          disable_tqdm=disable_tqdm,
                                          logging_steps=logging_steps,
                                          log_level="error",
                                          logging_dir="./logs",
                                          load_best_model_at_end=True,  # 在训练结束时加载最好的模型
                                          metric_for_best_model="accuracy")

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_data,
                      eval_dataset=test_data,
                      compute_metrics=compute_metrics,
                      tokenizer=tokenizer)

    if num_train_epochs > 0:
        trainer.train()

    evaluate_trainer(trainer, test_data, output_dir)

    # save model
    model.save_pretrained(f"{output}/model")