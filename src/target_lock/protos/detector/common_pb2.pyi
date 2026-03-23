from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoundingBox(_message.Message):
    __slots__ = ("x_min", "y_min", "x_max", "y_max")
    X_MIN_FIELD_NUMBER: _ClassVar[int]
    Y_MIN_FIELD_NUMBER: _ClassVar[int]
    X_MAX_FIELD_NUMBER: _ClassVar[int]
    Y_MAX_FIELD_NUMBER: _ClassVar[int]
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    def __init__(self, x_min: _Optional[float] = ..., y_min: _Optional[float] = ..., x_max: _Optional[float] = ..., y_max: _Optional[float] = ...) -> None: ...

class Point2D(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class Circle(_message.Message):
    __slots__ = ("center", "radius")
    CENTER_FIELD_NUMBER: _ClassVar[int]
    RADIUS_FIELD_NUMBER: _ClassVar[int]
    center: Point2D
    radius: float
    def __init__(self, center: _Optional[_Union[Point2D, _Mapping]] = ..., radius: _Optional[float] = ...) -> None: ...

class DetectionGeometry(_message.Message):
    __slots__ = ("box", "point", "circle", "custom")
    BOX_FIELD_NUMBER: _ClassVar[int]
    POINT_FIELD_NUMBER: _ClassVar[int]
    CIRCLE_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_FIELD_NUMBER: _ClassVar[int]
    box: BoundingBox
    point: Point2D
    circle: Circle
    custom: _struct_pb2.Struct
    def __init__(self, box: _Optional[_Union[BoundingBox, _Mapping]] = ..., point: _Optional[_Union[Point2D, _Mapping]] = ..., circle: _Optional[_Union[Circle, _Mapping]] = ..., custom: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class Detection(_message.Message):
    __slots__ = ("class_name", "class_id", "score", "geometry", "normalized_geometry")
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    CLASS_ID_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    GEOMETRY_FIELD_NUMBER: _ClassVar[int]
    NORMALIZED_GEOMETRY_FIELD_NUMBER: _ClassVar[int]
    class_name: str
    class_id: int
    score: float
    geometry: DetectionGeometry
    normalized_geometry: DetectionGeometry
    def __init__(self, class_name: _Optional[str] = ..., class_id: _Optional[int] = ..., score: _Optional[float] = ..., geometry: _Optional[_Union[DetectionGeometry, _Mapping]] = ..., normalized_geometry: _Optional[_Union[DetectionGeometry, _Mapping]] = ...) -> None: ...

class StreamConfig(_message.Message):
    __slots__ = ("stream_id", "video_codec", "width", "height", "score_threshold", "max_detections")
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    VIDEO_CODEC_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    stream_id: str
    video_codec: str
    width: int
    height: int
    score_threshold: float
    max_detections: int
    def __init__(self, stream_id: _Optional[str] = ..., video_codec: _Optional[str] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., score_threshold: _Optional[float] = ..., max_detections: _Optional[int] = ...) -> None: ...
