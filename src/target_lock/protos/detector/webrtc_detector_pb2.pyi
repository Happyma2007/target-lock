from target_lock.protos.detector import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class IceServer(_message.Message):
    __slots__ = ("urls", "username", "credential")
    URLS_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    urls: _containers.RepeatedScalarFieldContainer[str]
    username: str
    credential: str
    def __init__(self, urls: _Optional[_Iterable[str]] = ..., username: _Optional[str] = ..., credential: _Optional[str] = ...) -> None: ...

class SessionDescription(_message.Message):
    __slots__ = ("type", "sdp")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SDP_FIELD_NUMBER: _ClassVar[int]
    type: str
    sdp: str
    def __init__(self, type: _Optional[str] = ..., sdp: _Optional[str] = ...) -> None: ...

class IceCandidate(_message.Message):
    __slots__ = ("candidate", "sdp_mid", "sdp_mline_index")
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    SDP_MID_FIELD_NUMBER: _ClassVar[int]
    SDP_MLINE_INDEX_FIELD_NUMBER: _ClassVar[int]
    candidate: str
    sdp_mid: str
    sdp_mline_index: int
    def __init__(self, candidate: _Optional[str] = ..., sdp_mid: _Optional[str] = ..., sdp_mline_index: _Optional[int] = ...) -> None: ...

class CreateStreamRequest(_message.Message):
    __slots__ = ("config",)
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: _common_pb2.StreamConfig
    def __init__(self, config: _Optional[_Union[_common_pb2.StreamConfig, _Mapping]] = ...) -> None: ...

class CreateStreamReply(_message.Message):
    __slots__ = ("stream_id", "offer", "ice_servers")
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    OFFER_FIELD_NUMBER: _ClassVar[int]
    ICE_SERVERS_FIELD_NUMBER: _ClassVar[int]
    stream_id: str
    offer: SessionDescription
    ice_servers: _containers.RepeatedCompositeFieldContainer[IceServer]
    def __init__(self, stream_id: _Optional[str] = ..., offer: _Optional[_Union[SessionDescription, _Mapping]] = ..., ice_servers: _Optional[_Iterable[_Union[IceServer, _Mapping]]] = ...) -> None: ...

class StreamSignal(_message.Message):
    __slots__ = ("stream_id", "answer", "ice_candidate")
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    ICE_CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    stream_id: str
    answer: SessionDescription
    ice_candidate: IceCandidate
    def __init__(self, stream_id: _Optional[str] = ..., answer: _Optional[_Union[SessionDescription, _Mapping]] = ..., ice_candidate: _Optional[_Union[IceCandidate, _Mapping]] = ...) -> None: ...

class UpdateStreamReply(_message.Message):
    __slots__ = ("ice_candidates",)
    ICE_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    ice_candidates: _containers.RepeatedCompositeFieldContainer[IceCandidate]
    def __init__(self, ice_candidates: _Optional[_Iterable[_Union[IceCandidate, _Mapping]]] = ...) -> None: ...

class StreamDetectionsRequest(_message.Message):
    __slots__ = ("stream_id",)
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    stream_id: str
    def __init__(self, stream_id: _Optional[str] = ...) -> None: ...

class StreamDetectionsReply(_message.Message):
    __slots__ = ("stream_id", "request_id", "pts_ms", "detections", "frame_id")
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PTS_MS_FIELD_NUMBER: _ClassVar[int]
    DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    stream_id: str
    request_id: str
    pts_ms: int
    detections: _containers.RepeatedCompositeFieldContainer[_common_pb2.Detection]
    frame_id: int
    def __init__(self, stream_id: _Optional[str] = ..., request_id: _Optional[str] = ..., pts_ms: _Optional[int] = ..., detections: _Optional[_Iterable[_Union[_common_pb2.Detection, _Mapping]]] = ..., frame_id: _Optional[int] = ...) -> None: ...
