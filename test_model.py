from transformers import AutoTokenizer, AutoModelForCausalLM

model_path="./pretrained/Qwen2.5-0.5B"

tokenizer=AutoTokenizer.from_pretrained(model_path)
model=AutoModelForCausalLM.from_pretrained(model_path)

print("Model loaded successfully!")
print("Parameters:",sum(p.numel() for p in model.parameters()))

text="The future of artificial intelligence is"
inputs=tokenizer(text,return_tensors="pt")

outputs=model.generate(**inputs,max_new_tokens=30)

print(tokenizer.decode(outputs[0],skip_special_tokens=True))