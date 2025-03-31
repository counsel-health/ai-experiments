import litellm
litellm.drop_params = True

import dspy
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import List
import json
import openai
import os
from dotenv import load_dotenv
from typing import Literal
from dspy.evaluate import Evaluate
import datetime

# Define the prompt template
MEDICAL_CLASSIFICATION_PROMPT = """
You are a medical assistant. You are given a message from a patient. Categorize the message into one of the following categories:

- Medication/Treatment: The message mentions a specific medication, or is about medication or treatments in general
- Advice: The message is asking for medical advice 
- Diagnosis/Triage: The message is asking you to diagnose or triage a medical condition.
- Symptom: The message is describing a symptom that the patient is experiencing.
- Research: The message is asking for health-related research information or educational content
- Labs/Measurements: The message is asking about lab results or health measurements
- MedicalRecord: The message is about the patient's personal medical record.
- Other: The message does not fit into any of the above categories or is empty.

Use the exact string with no quotes as your output, and do not include any additional text.

Message: {{message}}
"""

# Define the valid categories
VALID_CATEGORIES = [
    "Medication/Treatment",
    "Advice",
    "Diagnosis/Triage",
    "Symptom",
    "Research",
    "Labs/Measurements",
    "MedicalRecord",
    "Other"
]



# Define the DSPy module for classification
# class MedicalMessageClassifier(dspy.Module):
#     def __init__(self):
#         super().__init__()
#         self.classifier = dspy.ChainOfThought(MedicalClassificationSignature)
    
#     def forward(self, message):
#         return self.classifier(message=message)

class MedicalClassificationSignature(dspy.Signature):
    """Signature for classifying medical messages into predefined categories."""
    message: str = dspy.InputField(desc="""You are a medical assistant. You are given a message from a patient. Categorize the message into one of the following categories
                                   Output must be one of the following categories:
- Medication/Treatment: The message mentions a specific medication, or is about medication or treatments in general
- Advice: The message is asking for medical advice 
- Diagnosis/Triage: The message is asking you to diagnose or triage a medical condition.
- Symptom: The message is describing a symptom that the patient is experiencing.
- Research: The message is asking for health-related research information or educational content
- Labs/Measurements: The message is asking about lab results or health measurements
- MedicalRecord: The message is about the patient's personal medical record.
- Other: The message does not fit into any of the above categories or is empty.""")
    category: Literal["Medication/Treatment", "Advice", "Diagnosis/Triage", "Symptom", "Research", "Labs/Measurements", "MedicalRecord", "Other"] = dspy.OutputField(
    )


# Define a dataset class for medical message classification
# class MedicalMessageExample(dspy.Example):
#     def __init__(self, message, category):
#         message = message
#         categories: List[str] = category.replace('"', '').split(',')
#         super().__init__(message=message, categories=categories)

# Define a metric to evaluate the classifier
def accuracy_metric(example: dspy.Example, prediction, trace=None):
    # print("trace: ", trace)
    return prediction.category in example.categories

# Load and prepare the dataset
def load_dataset(file_path):
    """
    Load dataset from a CSV file with 'message' and 'category' columns
    """
    df = pd.read_csv(file_path)
    examples = []  # Initialize the examples list
    for _, row in df.iterrows():
        input_data = json.loads(row['input'])
        message = input_data['message']
        expected = row['expected']
        categories: List[str] = expected.replace('"', '').split(',')
        example = dspy.Example(
            message=message,
            categories=categories
        )
        # print(example)
        examples.append(example.with_inputs("message"))
    return examples

def split_dataset(examples, train_ratio=0.7):
    """
    Split dataset into training and evaluation sets
    
    Args:
        examples: List of examples
        train_ratio: Ratio of examples to use for training (default: 0.7)
    
    Returns:
        train_data, eval_data: Tuple of training and evaluation datasets
    """
    import random
    random.shuffle(examples)  # Shuffle to ensure random split
    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]

def main():
    # Load environment variables from .env file
    load_dotenv()

    task_llm = dspy.LM(model="gpt-4o", provider="openai")
    prompt_gen_llm = dspy.LM(model="gpt-4o", provider="openai", temperature=1.0, max_tokens=10000)

    
    # Set up the language model with the API key
    dspy.settings.configure(lm=task_llm)
    
    # Load and split dataset
    all_data = load_dataset("medical_messages.csv")
    # print(all_data)
    train_data, eval_data = split_dataset(all_data, train_ratio=0.5)
    
    print(f"Dataset split: {len(train_data)} training examples, {len(eval_data)} evaluation examples")
    
    # Load the saved optimized classifier
    # saved_model_path = "saved/run_20250330_151302_gpt4o_light_predict/optimized_classifier.json"

    
    
    # Create a new classifier instance and restore its state
    # saved_classifier = dspy.ChainOfThought(MedicalClassificationSignature)
    # saved_classifier.load(saved_model_path)

    # test_result = saved_classifier(message=train_data[0].message)
    # print(test_result)

    # dspy.inspect_history()


    # return


    
    # Create the base classifier
    classifier = dspy.Predict(MedicalClassificationSignature)

    # test_example = train_data[0]
    # print(test_example)
    # result = classifier(message=test_example.message)
    # print(result)
    # print(classifier.dump_state())
    # return

    base_evaluator = Evaluate(devset=all_data, num_threads=10, display_progress=True, display_table=5)
    base_result = base_evaluator(classifier, metric=accuracy_metric)
    print("Base result on all data: ", base_result)

# # Evaluate the base classifier
    # print("Evaluating base classifier")
    evaluator = Evaluate(devset=all_data, num_threads=10, display_progress=True, display_table=5)
    # evaluator(classifier, metric=accuracy_metric)


    # Define the optimizer
    # optimizer = dspy.BootstrapFewShot(
    #     metric=lambda example, prediction, trace: accuracy_metric(example, prediction),
    #     max_bootstrapped_demos=30,
    #     max_labeled_demos=30,
    #     max_rounds=5
    # )
    optimizer = dspy.MIPROv2(metric=accuracy_metric, auto="heavy", prompt_model=prompt_gen_llm)

    optimized_classifier = optimizer.compile(
        classifier,
        trainset=train_data,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
        requires_permission_to_run=False,
    )
    
    print(optimized_classifier.dump_state())

    # Evaluate the optimized classifier
    print("Evaluating optimized classifier")
    result = evaluator(optimized_classifier, metric=accuracy_metric)
    print(result)
    
    # Create a timestamp and directory for saving
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./saved/run_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the evaluation result
    result_filename = os.path.join(save_dir, "evaluation_result.json")
    with open(result_filename, "w") as f:
        json.dump({"final_result": result, "base_result": base_result}, f, indent=2)
    print(f"Evaluation result saved to {result_filename}")
    
    # Save the optimized classifier
    model_filename = os.path.join(save_dir, "optimized_classifier.json")
    optimized_classifier.save(model_filename)
    print(f"Optimized classifier saved to {model_filename}")
    
    # Example of using the optimized classifier
    test_message = eval_data[0].message
    test_result = optimized_classifier(message=test_message)
    print(f"Test message: {test_message}")
    print(f"Predicted category: {test_result.category}")

    dspy.inspect_history(n=1)

if __name__ == "__main__":
    main()
