import pytest
import respx
import httpx
from llm4agents.transport.http import HttpTransport
from llm4agents.audio.audio import Audio
from llm4agents.errors import LLM4AgentsError


@pytest.fixture
def transport():
    return HttpTransport("https://api.example.com", "test-key", 5.0)


@pytest.fixture
def audio(transport):
    return Audio(transport)


@respx.mock
async def test_speech_create_returns_bytes_and_parsed_headers(audio):
    audio_bytes = b"\xff\xfb\x90\x00fake-mp3-bytes"
    respx.post("https://api.example.com/v1/audio/speech").mock(
        return_value=httpx.Response(
            200,
            content=audio_bytes,
            headers={
                "content-type": "audio/mpeg",
                "x-request-id": "req-123",
                "x-charged-usd-cents": "1",
                "x-model-used": "x-ai/grok-voice-tts-1.0",
            },
        )
    )
    result = await audio.speech.create(
        model="x-ai/grok-voice-tts-1.0",
        input="Hola, ¿cómo estás?",
        voice="sal",
        response_format="mp3",
    )
    assert result.data == audio_bytes
    assert result.content_type == "audio/mpeg"
    assert result.request_id == "req-123"
    assert result.charged_usd_cents == 1
    assert result.model_used == "x-ai/grok-voice-tts-1.0"

    sent = respx.calls.last.request
    import json as _json
    sent_body = _json.loads(sent.content)
    assert sent_body == {
        "model": "x-ai/grok-voice-tts-1.0",
        "input": "Hola, ¿cómo estás?",
        "voice": "sal",
        "response_format": "mp3",
    }


@respx.mock
async def test_speech_create_defaults_content_type_when_missing(audio):
    respx.post("https://api.example.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"raw-bytes")
    )
    result = await audio.speech.create(model="m", input="hi", voice="eve")
    assert result.content_type == "audio/mpeg"
    assert result.charged_usd_cents is None
    assert result.request_id is None
    assert result.model_used is None


@respx.mock
async def test_speech_create_402_raises_insufficient_balance(audio):
    respx.post("https://api.example.com/v1/audio/speech").mock(
        return_value=httpx.Response(
            402,
            json={"error": {"code": "insufficient_balance", "message": "Not enough balance"}},
        )
    )
    with pytest.raises(LLM4AgentsError) as exc_info:
        await audio.speech.create(model="m", input="hi", voice="sal")
    assert exc_info.value.code == "insufficient_balance"
    assert exc_info.value.status_code == 402


@respx.mock
async def test_speech_create_voice_passthrough(audio):
    respx.post("https://api.example.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"bytes")
    )
    await audio.speech.create(model="m", input="hi", voice="eve")
    sent = respx.calls.last.request
    import json as _json
    assert _json.loads(sent.content)["voice"] == "eve"
