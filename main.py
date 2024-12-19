import streamlit as st
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

# Configure page
st.set_page_config(page_title="LLM-Powered Prompt Generator")
st.title("LLM-Powered Prompt Generator")

# Environment variables for sensitive data
hf_token = ""
if not hf_token:
    st.error("Hugging Face token not found in environment variables")
    st.stop()

# Model configuration
try:
    login(hf_token)
    
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    FINETUNED_MODEL = "Shivank91/Prompt_Generator_Fine_Tuned"
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load model based on available hardware
        if device == "cuda":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            # CPU loading without quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map={"": device},
                torch_dtype=torch.float32,
            )
        
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
        return tokenizer, fine_tuned_model

    tokenizer, model = load_model()
    
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input interface
question = st.text_input("Enter your prompt:")

if question:
    if not question.strip():
        st.warning("Please enter a valid prompt")
        st.stop()
        
    try:
        with st.spinner("Generating response..."):
            inputs = tokenizer(question, return_tensors="pt").to(device)
            
            # Add memory efficiency for CPU
            if device == "cpu":
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'], 
                        max_length=1200, 
                        do_sample=True, 
                        temperature=0.7
                    )
            else:
                outputs = model.generate(
                    inputs['input_ids'], 
                    max_length=1200, 
                    do_sample=True, 
                    temperature=0.7
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(response)
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")