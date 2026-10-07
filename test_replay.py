from src.replay import get_replay_dataset

data=get_replay_dataset(max_samples=5)

for i,text in enumerate(data):
    print(f"\n--- Example {i+1} ---")
    print(text[:500])