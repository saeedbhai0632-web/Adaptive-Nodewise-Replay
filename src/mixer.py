def get_mixed_dataset(
    local_data,
    replay_data,
    replay_ratio=0.10,
    total_examples=200,
    seed=42
):
    import random

    if not 0.0<=replay_ratio<=1.0:
        raise ValueError("replay_ratio must be between 0 and 1")

    replay_examples=round(total_examples*replay_ratio)
    local_examples=total_examples-replay_examples

    if local_examples>len(local_data):
        raise ValueError("Not enough local data for requested training budget")

    if replay_examples>len(replay_data):
        raise ValueError("Not enough replay data for requested training budget")

    rng=random.Random(seed)

    local_indices=list(range(len(local_data)))
    replay_indices=list(range(len(replay_data)))

    rng.shuffle(local_indices)
    rng.shuffle(replay_indices)

    mixed_data=[]

    for i in range(local_examples):
        mixed_data.append(("local",local_data[local_indices[i]]))

    for i in range(replay_examples):
        mixed_data.append(("replay",replay_data[replay_indices[i]]))

    rng.shuffle(mixed_data)

    return mixed_data