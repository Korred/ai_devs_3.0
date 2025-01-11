from pathlib import Path
from enum import StrEnum
import jsonlines
import random

CORRECT_DATA_PATH = Path("./tasks/C04E02/lab_data/correct.txt")
INCORRECT_DATA_PATH = Path("./tasks/C04E02/lab_data/incorrect.txt")
FINE_TUNING_DATA_PATH = Path("./tasks/C04E02/fine_tuning")


class DataType(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"


def balance_datasets(set_a, set_b):
    # Balance two datasets by random oversampling the smaller set.

    if len(set_a) == len(set_b):
        return set_a, set_b
        
    if len(set_a) < len(set_b):
        smaller_set = set_a
        target_size = len(set_b)
    else:
        smaller_set = set_b
        target_size = len(set_a)
    
    # Random oversampling with replacement
    additional_samples = random.choices(
        smaller_set, 
        k=target_size - len(smaller_set)
    )
    balanced_set = smaller_set + additional_samples
    
    # Return in original order
    if len(set_a) < len(set_b):
        return balanced_set, set_b
    else:
        return set_a, balanced_set

def load_data(file_path: Path, data_type: DataType) -> list[dict]:
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append({
                "messages": [
                    {"role": "system", "content": "Classify the user provided message (combination of numbers) as TRUE or FALSE. Respond ONLY with the correct label."},
                    {"role": "user", "content": f"Is this TRUE or FALSE: {line.strip()}"},
                    {"role": "assistant", "content": data_type.value}
                ],
            })

    return data

def split_for_train_test(data: list[dict], train_ratio: float) -> tuple[list[dict], list[dict]]:
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]


correct = load_data(CORRECT_DATA_PATH, DataType.TRUE)
incorrect = load_data(INCORRECT_DATA_PATH, DataType.FALSE)


# Extract 80% for training and 20% for testing
correct_train, correct_test = split_for_train_test(correct, 0.8)
incorrect_train, incorrect_test = split_for_train_test(incorrect, 0.8)

# Balance the training and testing datasets
correct_train, incorrect_train = balance_datasets(correct_train, incorrect_train)
correct_test, incorrect_test = balance_datasets(correct_test, incorrect_test)

# Join the balanced datasets and shuffle
train_data = correct_train + incorrect_train
test_data = correct_test + incorrect_test

random.shuffle(train_data)
random.shuffle(test_data)

# Save the data to JSONL files for fine-tuning
with jsonlines.open(FINE_TUNING_DATA_PATH / "train.jsonl", "w") as writer:
    writer.write_all(train_data)

with jsonlines.open(FINE_TUNING_DATA_PATH / "test.jsonl", "w") as writer:
    writer.write_all(test_data)
