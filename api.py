import io
from io import BytesIO
import time
import os
import wave
import json
import numpy as np
import torch
from fastapi import FastAPI, Query, Depends, Response
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
import ChatTTS
import uvicorn

import argparse
from loguru import logger
from utils import split_text, combine_audio, pack_audio, batch_split

class TTS(BaseModel):
    """TTS GET/POST request"""
    text: str = Query("欢迎使用ChatTTS API", description="text to synthesize")
    spk_id: str = Query("default", description="speaker id")
    media_type: str = Query("wav", description="media type")


class Speaker(BaseModel):
    name: str = Query("", description="speaker name")


app = FastAPI()

chat = ChatTTS.Chat()
args = None

speaker = {}

def generate_tts_audio(text, spk_id = "default", max_length = 80):
    if spk_id is None or spk_id == "random" or speaker.get(spk_id) is None:
        print("load spk random")
        spk_emb = chat.sample_random_speaker()
        
        speed = 3
        oral = 0
        laugh = 0
        bk = 4
        top_P = 0.7
        top_K = 20
        temperature = 0.3
    else:
        print(f"load spk {spk_id}")
        spk_emb = speaker.get(spk_id)['emb']
        spk_data = speaker.get(spk_id)

        speed = max(spk_data.get("speed", 3), 0)
        oral = spk_data.get("oral", 0)
        laugh = spk_data.get("laugh", 0)
        bk = spk_data.get("break", 4)
        top_P = spk_data.get("top_P", 0.7)
        top_K = spk_data.get("top_K", 20)
        temperature = spk_data.get("temperature", 0.3)

    if oral < 0 or oral > 9 or laugh < 0 or laugh > 2 or bk < 0 or bk > 7:
        raise ValueError("oral_(0-9), laugh_(0-2), break_(0-7) out of range")

    params_infer_code = {
        'spk_emb': spk_emb,
        'prompt': f'[speed_{speed}]',
        'top_P': top_P,
        'top_K': top_K,
        'temperature': temperature,
    }

    refine_text_prompt = f"[oral_{oral}][laugh_{laugh}][break_{bk}]"
    params_refine_text = {
        'prompt': refine_text_prompt,
        'top_P': top_P,
        'top_K': top_K,
        'temperature': temperature,
    }

    start_time = time.time()
    all_wavs = []
    texts = split_text(text, max_length)

    batch_size = 4
    for batch in batch_split(texts, batch_size):
        wavs = chat.infer(batch, params_infer_code=params_infer_code, params_refine_text=params_refine_text, use_decoder=True, skip_refine_text=False)
        all_wavs.extend(wavs)

        torch.cuda.empty_cache()

    audio = combine_audio(all_wavs)
    audio = (audio * 32768).astype(np.int16)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Saving audio for spk_id = {spk_id} , took {elapsed_time:.2f}s")

    return audio

def tts_handle(params: TTS):
    logger.debug(params)

    audio_data = generate_tts_audio(params.text, params.spk_id, 120)
    sr = 24000
    audio_data = pack_audio(BytesIO(), audio_data, sr, params.media_type).getvalue()

    return Response(audio_data, media_type=f"audio/{params.media_type}")

@app.get("/")
async def index(params: TTS = Depends(TTS)):
    return tts_handle(params)


@app.post("/")
async def index_post(params: TTS):
    return tts_handle(params)

@app.get("/speaker")
async def speaker_list():
    l = list(speaker.keys())
    return {'code': 200, 'msg': 'success', 'data': l}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--spk-dir", type=str, default="./speaker_config")
    args = parser.parse_args()
    logger.debug(f"model_dir: {args.model_dir}")
    for file in os.listdir(args.spk_dir):
        if file.endswith(".json"):
            logger.debug(f"loading speaker config: {file[:-5]}")
            with open(os.path.join(args.spk_dir, file), 'r') as f:
                config = json.load(f)
            spk_name = file[:-5]
            if 'spk_model' in config:
                speaker[spk_name] = config
                speaker[spk_name]['emb'] = torch.load(config['spk_model'])
            else:
                logger.warning(f"Skipping speaker '{spk_name}' as 'spk_model' field is not found in the configuration.")
    chat.load_models(source="local", force_redownload=False, local_path=args.model_dir, compile=args.compile)
    uvicorn.run(app, host=args.host, port=args.port)
