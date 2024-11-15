from transformers import AutoModelForCausalLM, AutoTokenizer
base_model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer.save_pretrained("./base_model_tokenizer")
base_model.save_pretrained("./base_model")