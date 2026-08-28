"""Tests for device selection and the MPS conv2d workaround."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sound_event_detection.models.audioprotopnet_frame_detector import (
    _MPS_CONV_OUT_CHANNEL_LIMIT,
    _chunked_conv2d,
)
from sound_event_detection.utils.device import default_device, resolve_device


class TestResolveDevice:
    """`resolve_device` precedence: explicit argument > SED_DEVICE > auto."""

    def test_default_device_is_available(self) -> None:
        """`default_device` names a device torch can actually allocate on."""
        torch.zeros(1, device=default_device())

    def test_explicit_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit device overrides ``SED_DEVICE``."""
        monkeypatch.setenv("SED_DEVICE", "cuda")
        assert resolve_device("cpu") == "cpu"

    def test_env_wins_over_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``SED_DEVICE`` overrides auto-detection."""
        monkeypatch.setenv("SED_DEVICE", "cpu")
        assert resolve_device() == "cpu"

    def test_empty_env_falls_back_to_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty ``SED_DEVICE`` is ignored rather than used as a device."""
        monkeypatch.setenv("SED_DEVICE", "")
        assert resolve_device() == default_device()


class TestChunkedConv2d:
    """`_chunked_conv2d` matches `F.conv2d` on both sides of the MPS limit."""

    @pytest.mark.parametrize("out_channels", [8, _MPS_CONV_OUT_CHANNEL_LIMIT + 1])
    def test_matches_conv2d_on_cpu(self, out_channels: int) -> None:
        """On CPU the helper is a plain `F.conv2d`, whatever the channel count."""
        x = torch.randn(1, 4, 3, 5)
        weight = torch.randn(out_channels, 4, 1, 1)
        torch.testing.assert_close(_chunked_conv2d(x, weight), F.conv2d(x, weight))

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires an MPS device")
    def test_matches_cpu_above_mps_channel_limit(self) -> None:
        """Chunking repairs the wrong values plain `F.conv2d` returns on MPS.

        torch's MPS backend silently miscomputes conv2d above 2**16 output
        channels (https://github.com/pytorch/pytorch/issues/142836), which is
        the regime BirdCODE's prototype layer runs in.

        The final assertion is a canary: it fails once the upstream bug is
        fixed, at which point `_chunked_conv2d` can go. On a torch build that
        raises NotImplementedError instead of corrupting silently
        (https://github.com/pytorch/pytorch/issues/144445) that assertion is
        skipped -- chunking is still required either way.
        """
        torch.manual_seed(0)
        out_channels = _MPS_CONV_OUT_CHANNEL_LIMIT + 4096
        # The corruption needs a spatial extent comparable to the real
        # prototype layer's; a 1x1 output stays correct even above the limit.
        x = torch.randn(1, 64, 16, 38)
        weight = torch.randn(out_channels, 64, 1, 1)
        expected = F.conv2d(x, weight)

        chunked = _chunked_conv2d(x.to("mps"), weight.to("mps")).cpu()
        torch.testing.assert_close(chunked, expected)

        try:
            unchunked = F.conv2d(x.to("mps"), weight.to("mps")).cpu()
        except NotImplementedError:
            return  # this build guards the op instead of corrupting it
        assert not np.allclose(unchunked.numpy(), expected.numpy(), atol=1e-3), (
            "plain conv2d no longer misbehaves on MPS above the channel limit; "
            "the workaround in _chunked_conv2d may be removable"
        )
