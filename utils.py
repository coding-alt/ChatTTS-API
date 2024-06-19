import numpy as np
import io
from io import BytesIO
import soundfile as sf
import subprocess
from langchain_text_splitters import RecursiveCharacterTextSplitter

def combine_audio(wavs):
    """
    合并多段音频
    :param wavs:
    :return:
    """
    wavs = [normalize_audio(w) for w in wavs]  # 先对每段音频归一化
    combined_audio = np.concatenate(wavs, axis=1)  # 沿着时间轴合并
    return normalize_audio(combined_audio)  # 合并后再次归一化


def normalize_audio(audio):
    """
    Normalize audio array to be between -1 and 1
    :param audio: Input audio array
    :return: Normalized audio array
    """
    audio = np.clip(audio, -1, 1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def split_text(text, max_length = 80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=0,
        separators=[
            "\n\n", "\n", ".", "。", ",", "，", "、", ":", "：", ";", "；", "!", "！", "?", "？",
        ]
    )
    
    return text_splitter.split_text(text)

def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer

def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def batch_split(items, batch_size=5):
    """
    将items划分为大小为batch_size的批次
    :param items:
    :param batch_size:
    :return:
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]