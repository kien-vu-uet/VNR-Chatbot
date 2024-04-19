import os
from threading import Thread
from typing import Iterator, List, Tuple

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
SYS_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on previous responses of assistant." \
    "\nAs the intent analysis assistant, analyze and break down the script into single statements (if any) so that the software can execute sequentially. " \
    "Note: you only need to provide results with a similar layout to the output in the above examples, no other information is needed. " \
    "At the same time, pay attention to linking words and determine their role in separating sentences. You can refer to the examples below to be able to separate the required sentence."   
        
CHAT_HISTORY = [
    (
        "Vẽ hình tròn màu đen ở vị trí (20, 40) rồi tô màu đen cho nó.", 
        "- Vẽ hình tròn màu đen ở vị trí (20, 40).\n- Tô màu đen cho hình tròn ở vị trí (20, 40)"
    ),
    (
        "Vẽ hình vuông và hình tròn với cùng kích thước 40 x 40 tại 2 vị trí là (24, 14) và góc trên bên trái màn hình.",
        "- Vẽ hình vuông có kích thước 40 x 40 tại vị trí (24, 14).\n- Vẽ hình tròn có kích thước 40 x 40 tại vị trí góc trên bên trái màn hình."
    ),
    (
        "Xoay hình vuông có cạnh dài 50 pixel sang phải 15 độ rồi tô màu đen cho nó.",
        "- Xoay hình vuông có cạnh dài 50 pixel sang phải 15 độ.\n- Tô màu đen cho hình vuông có cạnh dài 50 pixel."
    ),
    (
        'Vẽ hình mũi tên có chiều rộng và chiều cao lần lượt là 70 và 30 ở chính giữa sau đó tạo bản sao của nó ở góc bên trái.',
        '- Vẽ hình mũi tên có chiều rộng và chiều cao lần lượt là 70 và 30 ở vị trí chính giữa.\n- Sao chép.\n- Dán vào góc bên trái'
    ),
    (
        'Vẽ hình vuông và hình tròn có cùng kích thước là 30 pixel ở trung tâm, ở phía góc trái đặt 1 hình mũi tên màu đen có kích thước 230x34',
        '- Vẽ hình vuông có kích thước là 30 pixel ở trung tâm.\n- Vẽ hình tròn có kích thước là 30 pixel ở trung tâm.\n- Ở phía góc trái đặt 1 hình mũi tên màu đen có kích thước 230x34.'
    ),
    (
        'Tạo hình bình hành màu xanh dương , có chiều dài đáy là 25 đơn vị , chiều cao là 35 đơn vị , nằm ở vị trí tương đối trên cùng bên phải.',
        '- Tạo hình bình hành màu xanh dương , có chiều dài đáy là 25 đơn vị , chiều cao là 35 đơn vị , nằm ở vị trí tương đối trên cùng bên phải.'
    ),
    (
        'Hãy vẽ 1 hình vuông ở góc bên trái có chiều rộng là 30 pixel đồng thời vẽ 1 hình tròn nằm bên trong nó có bán kính bằng 1/2 chiều rộng, sau đó đổ màu vàng cho chúng.',
        '- Vẽ hình vuông ở góc bên trái có chiều rộng là 30 pixel và có màu vàng.\n- Vẽ hình tròn ở góc bên trái có bán kính là 15 pixel và có màu vàng.'
    )
] 
        
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


if torch.cuda.is_available():
    # model_id = "vilm/vinallama-7b-chat"
    # model_id = "LR-AI-Labs/vbd-llama2-7B-50b-chat"
    model_id = "Viet-Mistral/Vistral-7B-Chat"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", cache_dir='/hf_cache')
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/hf_cache')
    tokenizer.use_default_system_prompt = False


@spaces.GPU
def generate(
    message: str,
    chat_history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id 
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6, value=SYS_PROMPT),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0.1,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=1.0,
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
            value=1.0,
        ),
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

with gr.Blocks(css="style.css") as demo:
    # gr.Markdown(DESCRIPTION)
    # gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    chat_interface.render()
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
