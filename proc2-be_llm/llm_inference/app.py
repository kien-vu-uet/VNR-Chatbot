import os
from threading import Thread
from typing import Iterator, List, Tuple

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import huggingface_hub
from gradio_client import Client

import requests, json

MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 512
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "2048"))
SYS_PROMPT = "Bạn là một trợ lý tư vấn pháp lý về các vấn đề Luật và Quy chuẩn - Tiêu chuẩn. "\
             "Hãy suy luận để phản hồi các yêu cầu của người dùng dựa trên tài liệu hoặc thông tin được cung cấp (nếu có). " \
             "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. " \
             "Nếu bạn không biết câu trả lời cho một câu hỏi hoặc không tìm thấy thông tin liên quan, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
            #  "Based on the information provided by the retrieval system and argue to answer the user's question" 
        
DESCRIPTION = """\
# Llama-2 7B Chat

This Space demonstrates model [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta, a Llama 2 model with 7B parameters fine-tuned for chat instructions. Feel free to play with it, or duplicate to run generations without a queue! If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://huggingface.co/inference-endpoints).

🔎 For more details about the Llama 2 family of models and how to use them with `transformers`, take a look [at our blog post](https://huggingface.co/blog/llama2).

🔨 Looking for an even more powerful model? Check out the [13B version](https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat) or the large [70B model demo](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI).
"""

LICENSE = """
<p/>

---
As a derivate work of [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/USE_POLICY.md).
"""

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

model_id = "Viet-Mistral/Vistral-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/hf_cache')
tokenizer.use_default_system_prompt = False
# if torch.cuda.is_available():
#     huggingface_hub.login(os.getenv('HF_TOKEN'))
#     # model_id = "capleaf/T-Llama"
#     # peft_model_id = "kien-vu-uet/T-Llama-sft-registration-domain"
#     model_id = "Viet-Mistral/Vistral-7B-Chat"
#     peft_model_id = "kien-vu-uet/Vistral-sft-registration-domain"
    
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )
#     model = AutoModelForCausalLM.from_pretrained(model_id, 
#                                                 #  torch_dtype=torch.bfloat16, 
#                                                  quantization_config=bnb_config,
#                                                  device_map="auto", 
#                                                  cache_dir='/hf_cache')
#     model.load_adapter(peft_model_id)



# @spaces.GPU
def generate(
    message: str,
    chat_history: List[Tuple[str, str]],
    input_file:str,
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    activate_chat_history: bool = False,
) -> Iterator[str]:
    conversation = [] 
    if system_prompt:
        system_prompt = SYS_PROMPT
        conversation.append({"role": "system", "content": system_prompt})
    if activate_chat_history:
        for user, assistant in chat_history:
            conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    if input_file:
        input = open(input_file, 'r').read()
        conversation.append({"role": "user", "content": input})
    else:
        conversation.append({"role": "user", "content": message})
        
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False) \
                        .replace(tokenizer.bos_token, '') \
                        .replace(tokenizer.eos_token, '')
    headers = {
        'Content-Type': 'application/json',
        'Authorization':  f"Bearer {os.getenv('LLM_SERVER_API_KEY')}"
    }
    json_data = {
        "prompt": prompt, 
        "n_predict": max_new_tokens,
        "seed": "-1",
        "temp": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "logit_bias": "38368-inf",
        "repeat_penalty": repetition_penalty,
        "repeat_last_n": 64,
        "stream": True
    }
    
    response = requests.post('http://llm-inference-platform-cuda:8080/completion', 
                             headers=headers, 
                             json=json_data, 
                             stream=True)

    outputs = []
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            try:
                text = json.loads(chunk.decode()[6:])['content']
                outputs.append(text)
                yield "".join(outputs)
            except Exception as e:
                print(e.args)
        
    # input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    
    # if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
    #     input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
    #     gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    # input_ids = input_ids.to(model.device)

    # streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    # generate_kwargs = dict(
    #     {"input_ids": input_ids},
    #     streamer=streamer,
    #     max_new_tokens=max_new_tokens,
    #     do_sample=True,
    #     top_p=top_p,
    #     top_k=top_k,
    #     temperature=temperature,
    #     num_beams=1,
    #     no_repeat_ngram_size=2,
    #     repetition_penalty=repetition_penalty,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.pad_token_id 
    # )
    # t = Thread(target=model.generate, kwargs=generate_kwargs)
    # t.start()

    # outputs = []
    # for text in streamer:
    #     outputs.append(text)
    #     yield "".join(outputs)
    # del input_ids
    # t.join()


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.File(
            label='Input file',
            file_count='single',
            file_types=['text'],
            type='filepath',
            value=None,
        ),
        gr.Textbox(label="System prompt", lines=6, value=SYS_PROMPT),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.3,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.05,
        ),
        gr.Checkbox(
            label='Activate chat history',
            value=False,
        )
    ],
    stop_btn="Stop",
    examples=[
        ["Hello there! How are you doing?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ],
)

with gr.Blocks(css="style.css", title='LLM-Inference') as demo:
    # gr.Markdown(DESCRIPTION)
    # gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    chat_interface.render()
    # for component in chat_interface.additional_inputs:
    #     component.visible = False
    # gr.Markdown(LICENSE)

# if __name__ == "__main__":    
    
#     # app = gr.mount_gradio_app(app, demo, path=ROUTE)
    
#     demo.queue(max_size=20, api_open=True).launch(
#         share=True, 
#         # auth=("nlplab", "abc123"), 
#         show_error=True,
#         show_api=True,
#         server_port=7860,
#         server_name='0.0.0.0',
#         debug=True
#         # share_server_address='0.0.0.0:7860'
#         )


# /workspace/nlplab/kienvt/KLTN/llama.cpp/main \
    # -m ./peft_model/Vistral-7B-quantized.gguf \
        # --seed "-1" \
            # -c 2048 \
                # -f prompt.txt \
                    # -n 512 \
                        # --temp 0.1 \
                            # --top-p 0.95 \
                                # --top-k 40 \
                                    # --logit-bias 38368-inf \
                                        # --repeat-penalty 1.05 \
                                            # --repeat-last-n 64 \
                                                # --log-disable \
                                                    # --color 