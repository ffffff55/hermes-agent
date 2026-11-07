"""MEDIA attachments follow the runner's current adapter after a reconnect.

Final text already goes out through ``_final_delivery_adapter`` (a reconnect can
swap the registry adapter mid-turn). ``_deliver_media_attachments`` used to send
every MEDIA file, image batch, and failure notice through ``self`` — the adapter
that received the message, which no longer has a client. The user got the text
and neither the file nor a notice (#150782).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _MediaAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform = Platform.DISCORD, *, document_error: str | None = None):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self.documents: list[str] = []
        self.image_batches: list[list] = []
        self.notices: list[tuple[str, object]] = []
        self.document_error = document_error

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.notices.append((content or "", metadata))
        return SendResult(success=True, message_id="notice")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def send_document(self, chat_id, file_path, caption=None, file_name=None,
                            reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.documents.append(file_path)
        if self.document_error:
            return SendResult(success=False, error=self.document_error)
        return SendResult(success=True, message_id="doc")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0.0) -> SendResult:
        self.image_batches.append(list(images))
        return SendResult(success=True, message_id="imgs")


def _event() -> MessageEvent:
    return MessageEvent(text="", source=SessionSource(platform=Platform.DISCORD, chat_id="thread-1"))


def _deliver(adapter, doc: str, image: str):
    results: list[SendResult] = []
    return adapter._deliver_media_attachments(
        _event(), [(doc, False), (image, False)], [], force_document_attachments=False,
        human_delay=0, metadata={"thread_id": "42"}, record_delivery=results.append), results


@pytest.mark.asyncio
async def test_reconnect_delivers_files_and_failure_notice_on_the_live_adapter(tmp_path):
    """The replaced adapter must not be the one that uploads or notifies."""
    doc = tmp_path / "notes.md"
    image = tmp_path / "shot.png"
    doc.write_text("hello")
    image.write_bytes(b"\x89PNG")
    dead = _MediaAdapter(document_error="Not connected")
    live = _MediaAdapter(document_error="upload rejected")
    runner = SimpleNamespace(_delivery_adapter_for=lambda source: live)
    dead.gateway_runner = live.gateway_runner = runner

    call, results = _deliver(dead, str(doc), str(image))
    await call

    assert dead.documents == []
    assert dead.image_batches == []
    assert dead.notices == []
    assert live.documents == [str(doc)]
    assert len(live.image_batches) == 1
    assert any(str(image) in url for url, _alt in live.image_batches[0])
    assert live.notices == [(
        "⚠️ Couldn't deliver the file attachment (notes.md).", {"thread_id": "42"})]
    assert [result.success for result in results] == [True, False]


@pytest.mark.asyncio
async def test_media_attachments_stay_on_the_caller_without_a_replacement(tmp_path):
    """No runner, the same adapter, or a different platform: nothing is delegated."""
    doc = tmp_path / "keep.md"
    image = tmp_path / "keep.png"
    doc.write_text("x")
    image.write_bytes(b"\x89PNG")

    alone = _MediaAdapter()
    call, results = _deliver(alone, str(doc), str(image))
    await call
    assert alone.documents == [str(doc)]
    assert alone.notices == []
    assert [result.success for result in results] == [True, True]

    same = _MediaAdapter()
    same.gateway_runner = SimpleNamespace(_delivery_adapter_for=lambda source: same)
    call, results = _deliver(same, str(doc), str(image))
    await call
    assert same.documents == [str(doc)]
    assert same.notices == []

    discord = _MediaAdapter()
    telegram = _MediaAdapter(platform=Platform.TELEGRAM)
    discord.gateway_runner = SimpleNamespace(_delivery_adapter_for=lambda source: telegram)
    call, _results = _deliver(discord, str(doc), str(image))
    await call
    assert discord.documents == [str(doc)]
    assert telegram.documents == []
    assert telegram.image_batches == []
