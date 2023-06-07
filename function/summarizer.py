from transformers import pipeline, AutoModel, AutoTokenizer, TFAutoModelForSeq2SeqLM, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

MODEL_DIR = "summarizer_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# def preprocess_function(examples):
#     inputs = [prefix + doc for doc in examples["text"]]
#     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

#     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)

#     return {k: round(v, 4) for k, v in result.items()}

# def training_model():
#     # Use labelled datasets
#     billsum = load_dataset("billsum", split="ca_test")

#     # split train and test set
#     billsum = billsum.train_test_split(test_size=0.2)

#     # load pre-trained t5-small model
#     checkpoint = "t5-small"
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#     # preprocess billsum datasets
#     prefix = "summarize: "
#     tokenized_billsum = billsum.map(preprocess_function, batched=True)

#     data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

#     # instantiate evaluate function
#     rouge = evaluate.load("rouge")

#     # Define arguments to train the model
#     training_args = Seq2SeqTrainingArguments(
#     output_dir="my_topic_summarizer_model",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=4,
#     predict_with_generate=True,
#     fp16=False,
#     push_to_hub=True,
#     )

#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_billsum["train"],
#         eval_dataset=tokenized_billsum["test"],
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()

#     # Push the model to Huggingface Repo
#     trainer.push_to_hub()


def inference_model(text):
    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, do_sample=False, min_length=3, max_length=200)
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text_output

def inference_all_data(data):
    feedback = data["feedback"]
    texts = "summarize: "
    for els in feedback:
        texts = texts + str(els) + ". "
    summarized_texts = {}

    # feedbacks = ["summarize: " + strings for strings in feedback]
    # text_outputs = list(map(inference_model, feedbacks))
    # summarized_texts["feedback"] = text_outputs

    summarized_text = inference_model(texts)
    summarized_text = [summarized_text]
    summarized_texts["feedback"] = summarized_text

    return summarized_texts


