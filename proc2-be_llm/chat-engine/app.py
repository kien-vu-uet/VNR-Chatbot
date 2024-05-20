import os, requests, json, re
from threading import Thread
from typing import Iterator, List, Tuple
from transformers import AutoTokenizer

import gradio as gr
from gradio_client import Client

username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')
RA_client = Client("http://rag-backend-server:5004/api/", auth=(username, password))
# LLM_client = Client("http://hf-llm-chatbot:5002/chat", auth=(username, password))

model_id = "Viet-Mistral/Vistral-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='/hf_cache')
tokenizer.use_default_system_prompt = False

MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "2048"))
SYS_PROMPT = "Bạn là một trợ lý tư vấn pháp lý về các vấn đề Luật và Quy chuẩn - Tiêu chuẩn. "\
             "Hãy suy luận để phản hồi các yêu cầu của người dùng dựa trên tài liệu hoặc thông tin được cung cấp (nếu có). " \
             "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. " \
             "Nếu bạn không biết câu trả lời cho một câu hỏi hoặc không tìm thấy thông tin liên quan, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
             
def prompt_format(system_prompt, instruction):
    prompt = f"""{system_prompt}
            ####### Instruction:
            {instruction}
            %%%%%%% Response:
            """
    return prompt

def generate(
    message: str,
    chat_history: List[Tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repeat_penalty: float = 1.2,
    repeat_last_n: int = 64,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    ra_outputs = RA_client.predict(query=message, expand_query=True, api_name="/search")
    context = "\n\n".join([item['_doc'] for item in ra_outputs])
    query = "Bạn hãy trả lời câu hỏi dựa vào thông tin sau:\n" + context + "\nCâu hỏi:" + message
    conversation.append({"role": "user", "content": query})
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
        "seed": -1,
        "temp": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "logit_bias": os.getenv('LOGIT_BIAS'),
        "repeat_penalty": repeat_penalty,
        "repeat_last_n": repeat_last_n,
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
                data = re.search(r'\{.*\}', chunk.decode('utf-8', 'ignore')).group()
                text = json.loads(data)['content']
                outputs.append(text)
                yield "".join(outputs)
            except Exception as e:
                print(chunk.decode('utf-8', 'ignore'), e.args)
                
                

chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
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
        gr.Slider(
            label="Repetition last N tokens",
            minimum=-1,
            maximum=400,
            step=1,
            value=100,
        ),
    ],
    examples=[
        ["Nguyên tắc trong giao thông đường sắt là gì?"],
        ["Quy định mới nhất về đăng kiểm phương tiện đường thuỷ nội địa?"],
        ["Quy trình đăng ký xe cơ giới"],
    ],
    submit_btn=gr.Button("💬 Gửi"),
    retry_btn=gr.Button("🔄 Thử lại"),
    stop_btn=gr.Button("⛔ Dừng"),
    clear_btn=gr.Button("🪄 Xóa"),
    undo_btn=gr.Button("💢 Quay lại"),
)

with gr.Blocks(css="style.css", title='VR-Chatbot') as demo:
    gr.Markdown('# Vietnam Register Chatbot')
    # gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    chat_interface.render()
    # for component in chat_interface.additional_inputs:
    #     component.visible = False
    # gr.Markdown(LICENSE)
