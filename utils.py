import re
import cn2an
import numpy as np
import jieba.posseg as pseg
from io import BytesIO
import soundfile as sf
import subprocess
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def remove_chinese_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！（），【】『』「」《》－‘“’”:,;!\(\)\[\]><\-] 替换为 ，
    :param text:
    :return:
    """
    chinese_punctuation_pattern = r"[：；！？（），【】『』「」《》－‘“’”:,;!\?\(\)\[\]><\-·]"
    text = re.sub(chinese_punctuation_pattern, '，', text)
    # 使用正则表达式将多个连续的句号替换为一个句号
    text = re.sub(r'[。，]{2,}', '。', text)
    # 删除开头和结尾的 ， 号
    text = re.sub(r'^，|，$', '', text)
    return text

def remove_english_punctuation(text):
    """
    移除文本中的中文标点符号 [：；！？（），【】『』「」《》－‘“’”:,;!\?\(\)\[\]><\-] 替换为 ，
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


def text_normalize(text):
    """
    对文本进行归一化处理
    :param text:
    :return:
    """
    from zh_normalization import TextNormalizer
    # ref: https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    # print(sentences)

    _txt = ''.join(sentences)
    # 替换掉除中文之外的所有字符
    # _txt = re.sub(
    #     r"[^\u4e00-\u9fa5，。！？、]+", "", _txt
    # )

    return _txt


def convert_numbers_to_chinese(text):
    """
    将文本中的数字转换为中文数字 例如 123 -> 一百二十三
    :param text:
    :return:
    """
    return cn2an.transform(text, "an2cn")


def detect_language(sentence):
    # ref: https://github.com/2noise/ChatTTS/blob/main/ChatTTS/utils/infer_utils.py#L55
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_word_pattern = re.compile(r'\b[A-Za-z]+\b')

    chinese_chars = chinese_char_pattern.findall(sentence)
    english_words = english_word_pattern.findall(sentence)

    if len(chinese_chars) > len(english_words):
        return "zh"
    else:
        return "en"


def split_text(text, max_length = 80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=0,
        separators=[
            "\n\n", "\n", ".", "。", ",", "，", "？", "?", "！", "!"
        ]
    )
    
    result = text_splitter.split_text(text)
    print("result", result)
    if detect_language(text[:1024]) == "zh":
        result = [normalize_zh(_.strip()) for _ in result if _.strip()]
    else:
        result = [normalize_en(_.strip()) for _ in result if _.strip()]
    return result


def normalize_en(text):
    from tn.english.normalizer import Normalizer
    normalizer = Normalizer()
    return remove_english_punctuation(normalizer.normalize(text))


def normalize_zh(text):
    return process_ddd(text_normalize(remove_chinese_punctuation(text)))

def process_ddd(text):
    """
    处理“地”、“得” 字的使用，都替换为“的”
    依据：地、得的使用，主要是在动词和形容词前后，本方法没有严格按照语法替换，因为时常遇到用错的情况。
    另外受 jieba 分词准确率的影响，部分情况下可能会出漏掉。例如：小红帽疑惑地问
    :param text: 输入的文本
    :return: 处理后的文本
    """
    word_list = [(word, flag) for word, flag in pseg.cut(text, use_paddle=False)]
    # print(word_list)
    processed_words = []
    for i, (word, flag) in enumerate(word_list):
        if word in ["地", "得"]:
            # Check previous and next word's flag
            # prev_flag = word_list[i - 1][1] if i > 0 else None
            # next_flag = word_list[i + 1][1] if i + 1 < len(word_list) else None

            # if prev_flag in ['v', 'a'] or next_flag in ['v', 'a']:
            if flag in ['uv', 'ud']:
                processed_words.append("的")
            else:
                processed_words.append(word)
        else:
            processed_words.append(word)

    return ''.join(processed_words)

def batch_split(items, batch_size=5):
    """
    将items划分为大小为batch_size的批次
    :param items:
    :param batch_size:
    :return:
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

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