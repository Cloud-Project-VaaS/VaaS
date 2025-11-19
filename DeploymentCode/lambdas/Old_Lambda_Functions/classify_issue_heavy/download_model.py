from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "vinaykarman/robertabase"
SAVE_DIRECTORY = "./model"

if __name__ == "__main__":
    print(f"Downloading model '{MODEL_NAME}' to '{SAVE_DIRECTORY}'...")
    
    # Download and save the tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIRECTORY)
    
    # Download and save the model files
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIRECTORY)
    
    print("Download complete.")