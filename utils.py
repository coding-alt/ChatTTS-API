import numpy as np
import re
from io import BytesIO
import soundfile as sf
import subprocess
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 将多个音频数组结合在一起，同时在音频段的接合处使用交叉淡入淡出技术，以避免在拼接时出现剪切噪音（如“滋滋声”）
def combine_audio(audio_arrays, crossfade_duration=0.1, rate=24000):
    """
    Combine audio arrays with crossfade to avoid clipping noise at the junctions.
    :param audio_arrays: List of audio arrays to combine
    :param crossfade_duration: Duration of the crossfade in seconds
    :param rate: Sample rate of the audio
    :return: Combined audio array
    """
    crossfade_samples = int(crossfade_duration * rate)
    combined_audio = np.array([], dtype=np.float32)

    for i in range(len(audio_arrays)):
        audio_arrays[i] = np.squeeze(audio_arrays[i])  # Ensure all arrays are 1D
        if i == 0:
            combined_audio = audio_arrays[i]  # Start with the first audio array
        else:
            # Apply crossfade between the end of the current combined audio and the start of the next array
            overlap = np.minimum(len(combined_audio), crossfade_samples)
            crossfade_end = combined_audio[-overlap:]
            crossfade_start = audio_arrays[i][:overlap]
            # Crossfade by linearly blending the audio samples
            t = np.linspace(0, 1, overlap)
            crossfaded = crossfade_end * (1 - t) + crossfade_start * t
            # Combine audio by replacing the end of the current combined audio with the crossfaded audio
            combined_audio[-overlap:] = crossfaded
            # Append the rest of the new array
            combined_audio = np.concatenate((combined_audio, audio_arrays[i][overlap:]))

    return combined_audio

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

def remove_some_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-] 替换为 ，
    :param text:
    :return:
    """
    chinese_punctuation_pattern = r"[：；！？（），【】『』「」《》－‘“’”:,;!\?\(\)\[\]><\-·]"
    text = re.sub(chinese_punctuation_pattern, ',', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'[,\.]{2,}', '.', text)
    # 删除开头和结尾的 ， 号
    text = re.sub(r'^,|,$', '', text)
    return text

def split_text(text, max_length = 80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=0,
        separators=[
            "\n\n", "\n", ".", "。", ",", "，",
        ]
    )

    text = remove_some_punctuation(text)
    
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