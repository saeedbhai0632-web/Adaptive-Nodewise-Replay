from src.mixer import get_mixed_dataset

data=get_mixed_dataset(
    max_local_samples=90,
    max_replay_samples=10,
    replay_ratio=0.10
)

local_count=0
replay_count=0

for source,text in data:
    if source=="local":
        local_count+=1
    else:
        replay_count+=1

print("Total:",len(data))
print("Local:",local_count)
print("Replay:",replay_count)
print("Replay ratio:",replay_count/len(data))

print("\nFirst 10 examples:")
for source,text in data[:10]:
    print(source,":",text[:100].replace("\n"," "))