"""Loopback dictation. Decode bounded PCM in memory and run MLX Whisper on-device."""
import asyncio
import io
import os
import struct
import wave
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

SAMPLE_RATE = 16_000
MAX_SECONDS = 120
MAX_BYTES = 44 + SAMPLE_RATE * 2 * MAX_SECONDS
MODEL_PATH = Path(__file__).resolve().parent.parent / 'data' / 'models' / 'whisper-small-mlx'


def decode_wav(recording):
    if len(recording) < 44 or recording[:4] != b'RIFF' or struct.unpack_from('<I', recording, 4)[0] != len(recording) - 8:
        raise HTTPException(422, 'The recording is not a complete WAV file.')
    try:
        with wave.open(io.BytesIO(recording), 'rb') as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, SAMPLE_RATE, 'NONE'):
                raise HTTPException(415, 'Dictation requires mono 16 kHz, 16-bit PCM WAV audio.')
            frames = audio.getnframes()
            if frames > SAMPLE_RATE * MAX_SECONDS:
                raise HTTPException(413, 'Recordings are limited to two minutes.')
            pcm = audio.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise HTTPException(422, 'The recording is not a supported WAV file.') from exc
    if len(pcm) != 2 * frames:
        raise HTTPException(422, 'The recording is incomplete.')
    if frames < SAMPLE_RATE // 4 or not any(pcm):
        raise HTTPException(422, 'No microphone audio was captured. Record at least a quarter second and try again.')
    return pcm


def transcribe(pcm, model_path):
    if not all((model_path / filename).is_file() for filename in ('config.json', 'weights.npz')):
        raise HTTPException(503, 'The local Whisper Small model is not installed. See the Abyss setup instructions.')
    # The MLX loader also accepts a Hub repo. Use only a verified local path
    # and prevent Hub access even if the local files disappear during setup.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    try:
        import mlx.core as mx
        import mlx_whisper
        import numpy as np
    except ImportError as exc:
        raise HTTPException(503, 'MLX Whisper is unavailable. Install the Abyss Python dependencies on Apple Silicon.') from exc

    waveform = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
    try:
        # Use an explicit GPU stream in the worker thread that runs inference.
        with mx.stream(mx.gpu):
            result = mlx_whisper.transcribe(
                waveform, path_or_hf_repo=str(model_path), verbose=None,
                temperature=0.0, condition_on_previous_text=False,
            )
    except Exception as exc:
        raise HTTPException(500, 'Whisper could not transcribe this recording. Try a shorter recording.') from exc
    text = result['text'].strip()
    if not text:
        raise HTTPException(422, 'No speech was recognized. Speak clearly and try again.')
    return text


def dictation_router():
    router = APIRouter()
    lock = asyncio.Lock()

    @router.post('/api/transcribe')
    async def record(request: Request):
        if request.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'audio/wav':
            raise HTTPException(415, 'Send microphone audio as a PCM WAV recording.')
        if lock.locked():
            raise HTTPException(409, 'Another recording is being transcribed. Try again in a moment.')
        async with lock:
            recording = bytearray()
            async for chunk in request.stream():
                if len(recording) + len(chunk) > MAX_BYTES:
                    raise HTTPException(413, 'The recording exceeds the two-minute or 3.84 MB limit.')
                recording.extend(chunk)
            pcm = decode_wav(recording)
            del recording
            text = await run_in_threadpool(transcribe, pcm, MODEL_PATH)
        return {'text': text}

    return router
