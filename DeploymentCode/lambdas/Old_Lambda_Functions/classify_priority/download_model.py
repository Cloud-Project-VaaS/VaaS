from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

priority_classes = ['high', 'medium', 'low']
save_directory = "models"
for priority in priority_classes:
    model_name = f"karths/modernbertbase-binary-10IQR-{priority}-priority"
    model_save_path = os.path.join(save_directory, priority)
    print(f"Downloading model for '{priority}' priority...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer.save_pretrained(model_save_path)
    model.save_pretrained(model_save_path)
print("All models downloaded.")