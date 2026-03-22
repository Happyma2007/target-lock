from __future__ import annotations

import unittest
from typing import Callable, Iterator

import numpy as np
from google.protobuf.struct_pb2 import Struct

from target_lock.protos.lockon import gym_env_pb2
from target_lock.sim.lockon import LockonSession, array_from_tensor


def _tensor(array: object, *, dtype: str | None = None) -> gym_env_pb2.Tensor:
    arr = np.asarray(array, dtype=dtype)
    return gym_env_pb2.Tensor(data=arr.tobytes(), shape=list(arr.shape), dtype=str(arr.dtype))


def _struct(**kwargs: object) -> Struct:
    msg = Struct()
    msg.update(kwargs)
    return msg


class _FakeChannel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeGrpc:
    def __init__(self, channel: _FakeChannel) -> None:
        self.channel = channel
        self.server_addr: str | None = None

    def insecure_channel(self, server_addr: str) -> _FakeChannel:
        self.server_addr = server_addr
        return self.channel


class _FakeStub:
    def __init__(
        self,
        channel: _FakeChannel,
        handler: Callable[[Iterator[gym_env_pb2.EnvRequest]], Iterator[gym_env_pb2.EnvReply]],
    ) -> None:
        self.channel = channel
        self.handler = handler

    def StreamEnv(self, request_iterator: Iterator[gym_env_pb2.EnvRequest]) -> Iterator[gym_env_pb2.EnvReply]:
        return self.handler(request_iterator)


class _FakeGrpcModule:
    def __init__(
        self,
        handler: Callable[[Iterator[gym_env_pb2.EnvRequest]], Iterator[gym_env_pb2.EnvReply]],
    ) -> None:
        self.handler = handler

    def GymEnvStub(self, channel: _FakeChannel) -> _FakeStub:
        return _FakeStub(channel, self.handler)


class _RecordingDecoder:
    def __init__(self, tensor_dtype: str, frame: np.ndarray) -> None:
        self.tensor_dtype = tensor_dtype
        self.frame = frame
        self.reset_calls = 0
        self.closed = False
        self.decode_calls: list[tuple[gym_env_pb2.Tensor, dict[str, object]]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True

    def decode(self, observation: gym_env_pb2.Tensor, info: dict[str, object]) -> np.ndarray:
        self.decode_calls.append((observation, dict(info)))
        return self.frame


def _stream_handler(
    requests: list[gym_env_pb2.EnvRequest],
    *,
    reset_reply: gym_env_pb2.EnvReply | None = None,
    step_reply: gym_env_pb2.EnvReply | None = None,
) -> Callable[[Iterator[gym_env_pb2.EnvRequest]], Iterator[gym_env_pb2.EnvReply]]:
    def _stream(request_iterator: Iterator[gym_env_pb2.EnvRequest]) -> Iterator[gym_env_pb2.EnvReply]:
        for request in request_iterator:
            requests.append(request)
            cmd = request.WhichOneof("cmd")
            if cmd == "reset":
                if reset_reply is None:
                    raise AssertionError("unexpected reset request")
                yield reset_reply
                continue
            if cmd == "step":
                if step_reply is None:
                    raise AssertionError("unexpected step request")
                yield step_reply
                continue
            if cmd == "close":
                yield gym_env_pb2.EnvReply(close=gym_env_pb2.CloseReply())
                return
            raise AssertionError(f"unexpected command {cmd!r}")

    return _stream


def _make_session(
    handler: Callable[[Iterator[gym_env_pb2.EnvRequest]], Iterator[gym_env_pb2.EnvReply]],
) -> tuple[LockonSession, _FakeChannel, _FakeGrpc]:
    channel = _FakeChannel()
    grpc = _FakeGrpc(channel)
    session = LockonSession(server_addr="127.0.0.1:50051")
    session.grpc = grpc
    session.gym_env_pb2_grpc = _FakeGrpcModule(handler)
    return session, channel, grpc


class LockonSessionTests(unittest.TestCase):
    def test_reset_decodes_v2_tensor_observation_with_reset_info(self) -> None:
        requests: list[gym_env_pb2.EnvRequest] = []
        reset_reply = gym_env_pb2.EnvReply(
            reset=gym_env_pb2.ResetReply(
                observation=gym_env_pb2.TensorValue(tensor=_tensor(np.zeros((2, 3, 3), dtype=np.uint8))),
                info=_struct(frame_codec="rgb", width=3, height=2),
            )
        )
        session, channel, grpc = _make_session(_stream_handler(requests, reset_reply=reset_reply))
        expected_frame = np.full((2, 3, 3), 7, dtype=np.uint8)
        decoder = _RecordingDecoder("uint8", expected_frame)
        seen_dtypes: list[str] = []

        def _create_decoder(tensor_dtype: str) -> _RecordingDecoder:
            seen_dtypes.append(tensor_dtype)
            return decoder

        session.create_observation_decoder = _create_decoder

        with session:
            frame = session.reset()

        np.testing.assert_array_equal(frame, expected_frame)
        self.assertEqual(seen_dtypes, ["uint8"])
        self.assertEqual(decoder.reset_calls, 1)
        self.assertEqual(decoder.decode_calls[0][1]["frame_codec"], "rgb")
        self.assertEqual(decoder.decode_calls[0][1]["width"], 3.0)
        self.assertEqual(requests[0].WhichOneof("cmd"), "reset")
        self.assertEqual(grpc.server_addr, "127.0.0.1:50051")
        self.assertTrue(channel.closed)
        self.assertTrue(decoder.closed)

    def test_step_wraps_action_as_tensor_value_and_decodes_single_env_scalars(self) -> None:
        requests: list[gym_env_pb2.EnvRequest] = []
        step_reply = gym_env_pb2.EnvReply(
            step=gym_env_pb2.StepReply(
                observation=gym_env_pb2.TensorValue(tensor=_tensor(np.zeros((4, 4, 3), dtype=np.uint8))),
                reward=[1.25],
                terminated=[True],
                truncated=[False],
                info=_struct(source="oracle", width=4),
            )
        )
        session, channel, _ = _make_session(_stream_handler(requests, step_reply=step_reply))

        with session:
            result = session.step(np.arange(6, dtype=np.float32))

        self.assertEqual(requests[0].WhichOneof("cmd"), "step")
        self.assertEqual(requests[0].step.action.WhichOneof("kind"), "tensor")
        np.testing.assert_array_equal(
            array_from_tensor(requests[0].step.action.tensor),
            np.arange(6, dtype=np.float32),
        )
        self.assertEqual(result.reward, 1.25)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.info["source"], "oracle")
        self.assertEqual(result.info["width"], 4.0)
        self.assertEqual(result.observation.dtype, "uint8")
        self.assertTrue(channel.closed)

    def test_reset_rejects_non_tensor_observation(self) -> None:
        requests: list[gym_env_pb2.EnvRequest] = []
        reset_reply = gym_env_pb2.EnvReply(
            reset=gym_env_pb2.ResetReply(
                observation=gym_env_pb2.TensorValue(
                    list=gym_env_pb2.TensorList(items=[_tensor([1.0], dtype="float32")])
                )
            )
        )
        session, _, _ = _make_session(_stream_handler(requests, reset_reply=reset_reply))

        with session:
            with self.assertRaisesRegex(RuntimeError, "reset.observation must be tensor-valued"):
                session.reset()

    def test_step_rejects_non_single_env_repeated_scalars(self) -> None:
        requests: list[gym_env_pb2.EnvRequest] = []
        step_reply = gym_env_pb2.EnvReply(
            step=gym_env_pb2.StepReply(
                observation=gym_env_pb2.TensorValue(tensor=_tensor(np.zeros((2, 2, 3), dtype=np.uint8))),
                reward=[0.0, 1.0],
                terminated=[False, False],
                truncated=[False, True],
            )
        )
        session, _, _ = _make_session(_stream_handler(requests, step_reply=step_reply))

        with session:
            with self.assertRaisesRegex(RuntimeError, "single-env repeated scalars"):
                session.step(np.zeros(6, dtype=np.float32))

    def test_step_rejects_non_single_env_action_shape(self) -> None:
        requests: list[gym_env_pb2.EnvRequest] = []
        session, _, _ = _make_session(_stream_handler(requests))

        with session:
            with self.assertRaisesRegex(ValueError, "single-env action shape \\[6\\]"):
                session.step(np.zeros((2, 6), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
