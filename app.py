import argparse
import os
import torch
from flask import Flask, request, jsonify
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.serve.utils import load_image, image_ext, video_ext
from videollava.utils import disable_torch_init
from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer

app = Flask(__name__)

# Global variables to store model and processors
model = None
tokenizer = None
image_processor = None
video_processor = None
conv_mode = None

def initialize_model(model_path, model_base, device="cuda", load_8bit=False, load_4bit=False):
    global model, tokenizer, image_processor, video_processor, conv_mode

    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device)
    image_processor, video_processor = processor['image'], processor['video']

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    file_paths = data.get('file_paths', [])
    text = data.get('text', '')
    temperature = data.get('temperature', 0.2)
    max_new_tokens = data.get('max_new_tokens', 512)

    tensor = []
    special_token = []
    for file in file_paths:
        if os.path.splitext(file)[-1].lower() in image_ext:
            file = image_processor.preprocess(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
            special_token += [DEFAULT_IMAGE_TOKEN]
        elif os.path.splitext(file)[-1].lower() in video_ext:
            file = video_processor(file, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)
            special_token += [DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames
        else:
            return jsonify({"error": f'Unsupported file type: {os.path.splitext(file)[-1].lower()}'}), 400
        tensor.append(file)

    conv = conv_templates[conv_mode].copy()
    roles = ('user', 'assistant') if "mpt" in model.config.model_type.lower() else conv.roles

    if getattr(model.config, "mm_use_im_start_end", False):
        inp = ''.join([DEFAULT_IM_START_TOKEN + i + DEFAULT_IM_END_TOKEN for i in special_token]) + '\n' + text
    else:
        inp = ''.join(special_token) + '\n' + text

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    
    return jsonify({"response": outputs})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    initialize_model(args.model_path, args.model_base, device=args.device, load_8bit=args.load_8bit, load_4bit=args.load_4bit)
    app.run(debug=True)