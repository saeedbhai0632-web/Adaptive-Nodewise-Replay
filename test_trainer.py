from transformers import AutoModelForCausalLM

from src.data import get_dataset
from src.replay import get_replay_dataset
from src.mixer import get_mixed_dataset
from src.trainer import train

MODEL_PATH="./pretrained/Qwen2.5-0.5B"

local_data=get_dataset(1,20)
replay_data=get_replay_dataset(20)

mixed_data=get_mixed_dataset(
    local_data,
    replay_data,
    replay_ratio=0.10
)

model=AutoModelForCausalLM.from_pretrained(MODEL_PATH)

model=train(
    model,
    mixed_data,
    steps=2
)

print("Training test completed.")