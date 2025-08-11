from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import torch
def get_llm(model_id="Qwen/Qwen2.5-7B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    hf_pipeline = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length = 300,
        device_map=0
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)

if __name__ == "__main__":
    llm = None
    if not llm:
        llm = get_llm()

    prompt = "1 sum 1 equal = ?"
    print(llm(prompt))
