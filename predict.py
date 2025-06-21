import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline,AutoConfig
from peft import PeftModel
import torch
import csv

model_path="/data/download-model/DeepSeek-R1-0528-Qwen3-8B"
lora_path="finetuned_model"

# 加载模型与分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
config.pad_token_id = tokenizer.pad_token_id  # 关键步骤
config.num_labels = 8
model = AutoModelForSequenceClassification.from_pretrained(model_path,config=config)
model = PeftModel.from_pretrained(model, lora_path)

model = model.merge_and_unload()

classifer = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

sentences=[]
with open("data/test.csv","r",encoding="utf-8") as f:
    reader=csv.reader(f)
    next(reader)
    for line in reader:
        sentences.append(line[1])

outputs=classifer(sentences)

results=[]

for idx,output in enumerate(outputs):
    result=[]
    result.append(idx)
    result.append(output['label'].split('_')[-1])
    results.append(result)

with open('submission.csv','w') as f:
    f.write('id,label\n')
    writer=csv.writer(f)
    writer.writerows(results)





