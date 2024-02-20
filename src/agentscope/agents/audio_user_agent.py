# -*- coding: utf-8 -*-
"""User Agent class"""
from typing import Union
from typing import Optional
from typing import Iterable
import uuid
import dashscope
import pyaudio
import wave
from webrtcvad import Vad
from dashscope.audio.asr import (
    Recognition,
    RecognitionCallback,
)

from agentscope.agents import AgentBase
from agentscope.message import Msg


def is_speech(vad: Vad, frame: bytes, rate: int) -> bool:
    """Whether the frame is speech."""
    return vad.is_speech(frame, rate)


def save_audio(
    buffer: Iterable,
    channels: int,
    rate: int,
    audio_path: str,
) -> None:
    """Save audio into file"""
    wf = wave.open(audio_path, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b"".join(buffer))
    wf.close()


def record_audio(
    vad: Vad,
    audio_path: str,
    channels: int = 1,
    rate: int = 16000,
    chunk: int = 480,
    max_interval: int = 1,
) -> None:
    """Record audio."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )
    print("start to record...")

    audio_buffer = []
    silence_frames = 0
    detected_speech = False

    while True:
        frame = stream.read(chunk)
        is_speech_frame = is_speech(vad, frame, rate)

        if detected_speech or is_speech_frame:
            audio_buffer.append(frame)

        if is_speech_frame:
            detected_speech = True
            silence_frames = 0
        elif detected_speech:
            silence_frames += 1

        if detected_speech and silence_frames > max_interval * rate / chunk:
            break

    print("Stop recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    save_audio(audio_buffer, channels, rate, audio_path)


def audio2text(audio_path: str) -> str:
    """Convert audio to text."""
    # dashscope.api_key = ""
    callback = RecognitionCallback()
    rec = Recognition(
        model="paraformer-realtime-v1",
        format="wav",
        sample_rate=16000,
        callback=callback,
    )

    result = rec.call(audio_path)
    return " ".join([s["text"] for s in result["output"]["sentence"]])


class AudioUserAgent(AgentBase):
    """Audio user agent class"""

    def __init__(self, name: str = "User", require_url: bool = False) -> None:
        """Initialize a UserAgent object.

        Arguments:
            name (`str`, defaults to `"User"`):
                The name of the agent. Defaults to "User".
            require_url (`bool`, defaults to `False`):
                Whether the agent requires user to input a URL. Defaults to
                False. The URL can lead to a website, a file,
                or a directory. It will be added into the generated message
                in field `url`.
        """
        super().__init__(name=name)

        self.name = name
        self.require_url = require_url
        self.vad = Vad(1)

    def reply(
        self,
        x: dict = None,
        required_keys: Optional[Union[list[str], str]] = None,
    ) -> dict:
        """
        Processes the input provided by the user and stores it in memory,
        potentially formatting it with additional provided details.

        The method prompts the user for input, then optionally prompts for
        additional specifics based on the provided format keys. All
        information is encapsulated in a message object, which is then
        added to the object's memory.

        Arguments:
            x (`dict`, defaults to `None`):
                A dictionary containing initial data to be added to memory.
                Defaults to None.
            required_keys \
                (`Optional[Union[list[str], str]]`, defaults to `None`):
                Strings that requires user to input, which will be used as
                the key of the returned dict. Defaults to None.

        Returns:
            `dict`: A dictionary representing the message object that contains
            the user's input and any additional details. This is also
            stored in the object's memory.
        """
        if x is not None:
            self.memory.add(x)

        audio_path = f"user-{uuid.uuid4()}.wav"
        record_audio(vad=self.vad, audio_path=audio_path)
        content = audio2text(audio_path=audio_path)
        kwargs = {}
        if required_keys is not None:
            if isinstance(required_keys, str):
                required_keys = [required_keys]

            for key in required_keys:
                kwargs[key] = input(f"{key}: ")

        url = None
        if self.require_url:
            url = input("URL: ")

        # Add additional keys
        msg = Msg(
            self.name,
            content=content,
            url=url,
            **kwargs,  # type: ignore[arg-type]
        )
        self.speak(msg)
        # Add to memory
        self.memory.add(msg)

        return msg
