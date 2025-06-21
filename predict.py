import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline,AutoConfig
from peft import PeftModel
import torch
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="训练模型的命令行参数")
    
    # 添加命令行参数
    parser.add_argument('--model_path', type=str, default="/data/download-model/DeepSeek-R1-0528-Qwen3-8B",
                        help='模型路径')
    parser.add_argument('--input_file', type=str, default="data/train.csv",
                        help='测试集')
    parser.add_argument('--output_file', type=str, default="submission.csv",
                        help='输出文件')
    parser.add_argument('--lora_path', type=str, default="finetuned_model",
                        help='lora路径')
    
    # 解析参数
    args = parser.parse_args()
    return args

args=parse_args()

model_path=args.model_path
lora_path=args.lora_path
input_file=args.input_file
output_file=args.output_file

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
with open(input_file,"r",encoding="utf-8") as f:
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

with open(output_file,'w') as f:
    f.write('id,label\n')
    writer=csv.writer(f)
    writer.writerows(results)





