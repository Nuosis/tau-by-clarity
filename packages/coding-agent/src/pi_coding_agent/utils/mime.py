"""MIME sniffing helpers."""
from __future__ import annotations

IMAGE_TYPE_SNIFF_BYTES = 4100
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _starts_with(buffer: bytes, prefix: bytes) -> bool:
    return buffer.startswith(prefix)


def _starts_with_ascii(buffer: bytes, offset: int, text: str) -> bool:
    return len(buffer) >= offset + len(text) and buffer[offset : offset + len(text)] == text.encode("ascii")


def _read_uint32_be(buffer: bytes, offset: int) -> int:
    value = buffer[offset : offset + 4]
    if len(value) < 4:
        value = value.ljust(4, b"\0")
    return int.from_bytes(value, "big")


def _is_png(buffer: bytes) -> bool:
    return len(buffer) >= 16 and _read_uint32_be(buffer, len(PNG_SIGNATURE)) == 13 and _starts_with_ascii(buffer, 12, "IHDR")


def _is_animated_png(buffer: bytes) -> bool:
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(buffer):
        chunk_length = _read_uint32_be(buffer, offset)
        chunk_type_offset = offset + 4
        if _starts_with_ascii(buffer, chunk_type_offset, "acTL"):
            return True
        if _starts_with_ascii(buffer, chunk_type_offset, "IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(buffer):
            return False
        offset = next_offset
    return False


def detect_supported_image_mime_type(buffer: bytes | bytearray | memoryview) -> str | None:
    raw = bytes(buffer)
    if _starts_with(raw, b"\xff\xd8\xff"):
        return None if len(raw) > 3 and raw[3] == 0xF7 else "image/jpeg"
    if _starts_with(raw, PNG_SIGNATURE):
        return "image/png" if _is_png(raw) and not _is_animated_png(raw) else None
    if _starts_with_ascii(raw, 0, "GIF"):
        return "image/gif"
    if _starts_with_ascii(raw, 0, "RIFF") and _starts_with_ascii(raw, 8, "WEBP"):
        return "image/webp"
    return None


def detect_supported_image_mime_type_from_file(file_path: str) -> str | None:
    with open(file_path, "rb") as handle:
        return detect_supported_image_mime_type(handle.read(IMAGE_TYPE_SNIFF_BYTES))


__all__ = [
    "IMAGE_TYPE_SNIFF_BYTES",
    "PNG_SIGNATURE",
    "detect_supported_image_mime_type",
    "detect_supported_image_mime_type_from_file",
]
