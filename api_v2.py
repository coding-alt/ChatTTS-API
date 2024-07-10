
import os
import uuid
import time
import json
import torch
import torchaudio
import traceback
import requests
import uvicorn
import ChatTTS
import argparse
from typing import Optional
from redis import Redis
from loguru import logger
from pydantic import BaseModel
from celery import Celery, states
from fastapi import FastAPI, Query, Response
from resemble_enhance.enhancer.inference import denoise, enhance
from utils import split_text, combine_audio, pack_audio, batch_split, save_audio

from volcengine_upload import upload_file_to_tos

class TTS(BaseModel):
    """TTS GET/POST request"""
    text: str = Query("欢迎使用ChatTTS API", description="text to synthesize")
    spk_id: str = Query("default", description="音色ID")
    group_id: str = Query("aeyes", description="group id")
    callback: Optional[str] = Query(None, description="callback function")

app = FastAPI()

# Celery配置
broker_url = "amqp://rabbitmq:rabbitmq@127.0.0.1:5672//?heartbeat=300"
backend_url = "redis://:redis@127.0.0.1:6379/0?socket_timeout=3"
result_store_url = "redis://:redis@127.0.0.1:6379/1?socket_timeout=3"

celery_app = Celery('api_v2', broker=broker_url, backend=backend_url)
celery_app.conf.update(
    task_always_eager=False,
    task_store_eager_result=False,
    result_expires=3600,
    task_default_retry_delay=5,
    task_soft_time_limit=3600
)

redis_client = Redis.from_url(result_store_url)

chat = ChatTTS.Chat()
args = None
speaker = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

def sync_request(url, method='GET', params=None, data=None, json=None, headers=None, retries=3):
    for i in range(retries):
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.text
            elif method == 'POST':
                response = requests.post(url, params=params, data=data, json=json, headers=headers)
                print(f"url: {url}, params: {params}, data: {data}, json: {json}, response: {response.text}")
                response.raise_for_status()
                return response.text
            else:
                raise ValueError(f"Unsupported method: {method}")
        except requests.RequestException as e:
            if i == retries - 1:
                raise e
            else:
                time.sleep(1)

def generate_tts_audio(text, spk_id="default", max_length=80, batch_size=4, denoise_audio = True, enhance_audio = True):
    audio_save_path = None
    try:
        if spk_id is None or spk_id == "random" or speaker.get(spk_id) is None:
            print("load spk random")
            spk_emb = chat.sample_random_speaker()
            speed, oral, laugh, bk, top_P, top_K, temperature = 3, 0, 0, 4, 0.7, 20, 0.3
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

        for batch in batch_split(texts, batch_size):
            wavs = chat.infer(batch, params_infer_code=params_infer_code, params_refine_text=params_refine_text, use_decoder=True, skip_refine_text=False)
            all_wavs.extend(wavs)
            torch.cuda.empty_cache()

        audio = combine_audio(all_wavs)

        os.makedirs('./result', exist_ok=True)
        audio_save_path = os.path.join('./result', f"{str(uuid.uuid4())}.wav")
        save_audio(audio_save_path, audio, rate=24000)

        dwav, sr = torchaudio.load(audio_save_path)
        dwav = dwav.mean(dim=0)

        if denoise_audio:
            dwav, sr = denoise(dwav, sr, device)

        if enhance_audio:
            nfe, solver, tau = 64, 'midpoint', 0.5

            # 执行增强
            enhanced_wav, enhanced_sample_rate = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.9 if denoise_audio else 0.1, tau=tau)

            enhanced_audio_data = enhanced_wav.cpu().numpy()
            audio_save_path = os.path.join('./result', f"{uuid.uuid4()}.wav")
            torchaudio.save(audio_save_path, torch.tensor(enhanced_audio_data).unsqueeze(0), enhanced_sample_rate)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Saving audio for spk_id = {spk_id} , took {elapsed_time:.2f}s")

        # 推送到对象火山引擎对象存储
        audio_url = upload_file_to_tos(audio_save_path)

        return audio_url, None
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error running task: {error_details}")
        return None, error_details
    finally:
        # 删除本地临时文件 
        if audio_save_path and os.path.exists(audio_save_path):
            os.remove(audio_save_path)

@celery_app.task(bind=True, max_retries=3, queue="tts_task")
def tts_handle(self, text, spk_id, group_id, callback):
    audio_url, msg = generate_tts_audio(text, spk_id)
    task_id = self.request.id
    if audio_url is not None: 
        result = {
            'task_id': task_id,
            'state': 'SUCCESS',
            'data': {
                'task_id': task_id,
                'audio_url': audio_url
            }
        }
    else:
        result = {
            'task_id': task_id,
            'state': 'FAILURE',
            'msg': msg
        }

    # 结果写入redis队列
    result_serialized = json.dumps(result)
    redis_client.rpush(group_id, result_serialized)

    if callback is not None:
        sync_request(callback, 'POST', data=result)

    return result

@app.post("/generate")
async def generate(params: TTS):
    try:
        task = tts_handle.apply_async(args=[params.text, params.spk_id, params.group_id, params.callback])
        # 更新任务状态为自定义状态：REAL_PENDING
        celery_app.backend.store_result(task_id=task.id, result=None, state='REAL_PENDING')
        return {"code": 200, "msg": "SUCCESS", "data": {"task_id": task.id}}
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error running task: {error_details}")
        return {"code": 500, "msg": error_details}

@app.get("/speaker")
async def speaker_list():
    l = list(speaker.keys())
    return {'code': 200, 'msg': 'success', 'data': l}

@app.post("/get_task_info")
async def get_task_info(task_id: str):
    if not task_id:
        return {'code': 400, 'msg': 'task_id is required'}
    
    task = celery_app.AsyncResult(task_id)
    if task.state == 'REAL_PENDING':
        return {'code': 200, 'msg': 'success', 'data': {'state': 'REAL_PENDING'}}
    elif task.state == states.STARTED:
        return {'code': 200, 'msg': 'success', 'data': {'state': 'STARTED'}}
    elif task.state == states.FAILURE:
        return {'code': 500, 'msg': task.traceback}
    elif task.state == states.SUCCESS:
        return {'code': 200, 'msg': 'success', 'data': {'state': 'SUCCESS', 'result': task.result}}
    else:
        return {'code': 200, 'msg': 'success', 'data': {'state': 'NOT_FOUND'}}

@app.get("/get_task_count")
def get_pending_task_counts():
    i = celery_app.control.inspect()
    active_tasks = i.active()
    task_count = len(active_tasks) if active_tasks else 0
    return {'code': 200, 'msg': 'success', 'data': {'task_count': task_count}}

@app.get("/get_group_results")
async def get_group_results(group_id: str, limit: int = 10):
    task_results = []
    for _ in range(limit):
        result_raw = redis_client.lpop(group_id)
        if not result_raw:
            break
        try:
            result_data = json.loads(result_raw.decode('utf-8'))
            task_results.append(result_data)
        except json.JSONDecodeError:
            continue

    return {"state": "SUCCESS", "results": task_results}

model_dir = "./models"
spk_dir = "./speaker_config"

for file in os.listdir(spk_dir):
    if file.endswith(".json"):
        logger.debug(f"loading speaker config: {file[:-5]}")
        with open(os.path.join(spk_dir, file), 'r') as f:
            config = json.load(f)
        spk_name = file[:-5]
        if 'spk_model' in config:
            speaker[spk_name] = config
            speaker[spk_name]['emb'] = torch.load(config['spk_model'])
        else:
            logger.warning(f"Skipping speaker '{spk_name}' as 'spk_model' field is not found in the configuration.")
chat.load_models(source="local", force_redownload=False, local_path=model_dir, compile=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)