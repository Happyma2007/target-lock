from target_lock.protos.detector import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ImageFrame(_message.Message):
    __slots__ = ("data", "width", "height", "format")
    DATA_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    width: int
    height: int
    format: str
    def __init__(self, data: _Optional[bytes] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., format: _Optional[str] = ...) -> None: ...

class DetectRequest(_message.Message):
    __slots__ = ("request_id", "frame", "score_threshold", "max_detections")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    frame: ImageFrame
    score_threshold: float
    max_detections: int
    def __init__(self, request_id: _Optional[str] = ..., frame: _Optional[_Union[ImageFrame, _Mapping]] = ..., score_threshold: _Optional[float] = ..., max_detections: _Optional[int] = ...) -> None: ...

class DetectReply(_message.Message):
    __slots__ = ("request_id", "detections")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    detections: _containers.RepeatedCompositeFieldContainer[_common_pb2.Detection]
    def __init__(self, request_id: _Optional[str] = ..., detections: _Optional[_Iterable[_Union[_common_pb2.Detection, _Mapping]]] = ...) -> None: ...

class EncodedVideoChunk(_message.Message):
    __slots__ = ("data", "pts_ms", "dts_ms", "key_frame", "codec")
    DATA_FIELD_NUMBER: _ClassVar[int]
    PTS_MS_FIELD_NUMBER: _ClassVar[int]
    DTS_MS_FIELD_NUMBER: _ClassVar[int]
    KEY_FRAME_FIELD_NUMBER: _ClassVar[int]
    CODEC_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    pts_ms: int
    dts_ms: int
    key_frame: bool
    codec: str
    def __init__(self, data: _Optional[bytes] = ..., pts_ms: _Optional[int] = ..., dts_ms: _Optional[int] = ..., key_frame: bool = ..., codec: _Optional[str] = ...) -> None: ...

class StreamDetectRequest(_message.Message):
    __slots__ = ("config", "frame", "chunk")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    config: _common_pb2.StreamConfig
    frame: ImageFrame
    chunk: EncodedVideoChunk
    def __init__(self, config: _Optional[_Union[_common_pb2.StreamConfig, _Mapping]] = ..., frame: _Optional[_Union[ImageFrame, _Mapping]] = ..., chunk: _Optional[_Union[EncodedVideoChunk, _Mapping]] = ...) -> None: ...

class StreamDetectReply(_message.Message):
    __slots__ = ("stream_id", "request_id", "pts_ms", "detections")
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PTS_MS_FIELD_NUMBER: _ClassVar[int]
    DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    stream_id: str
    request_id: str
    pts_ms: int
    detections: _containers.RepeatedCompositeFieldContainer[_common_pb2.Detection]
    def __init__(self, stream_id: _Optional[str] = ..., request_id: _Optional[str] = ..., pts_ms: _Optional[int] = ..., detections: _Optional[_Iterable[_Union[_common_pb2.Detection, _Mapping]]] = ...) -> None: ...
