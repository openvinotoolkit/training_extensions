# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from loguru import logger

from app.models.webrtc import Answer, InputData, Offer

from .broadcaster import FrameBroadcaster
from .sdp_handler import SDPHandler
from .stream import InferenceVideoStreamTrack


@dataclass(frozen=True)
class WebRTCSettings:
    config: RTCConfiguration
    advertise_ip: str | None = None


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    def __init__(self, frame_broadcaster: FrameBroadcaster, settings: WebRTCSettings, sdp_handler: SDPHandler) -> None:
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._input_data: dict[str, Any] = {}
        self._frame_broadcaster = frame_broadcaster
        self._settings = settings
        self._sdp_handler = sdp_handler
        self._lock = asyncio.Lock()

    async def handle_offer(self, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        pc = RTCPeerConnection(configuration=self._settings.config)

        async with self._lock:
            # If a connection already exists for this id, remember it so we can close it
            # *after* installing the new one. We must not unregister the broadcaster
            # consumer here: the new connection reuses the same id and the stale
            # connection's "closed" handler is identity-guarded in cleanup_connection.
            old_pc = self._pcs.get(offer.webrtc_id)
            self._pcs[offer.webrtc_id] = pc

        # Close any pre-existing connection for this id (outside the lock to avoid blocking).
        if old_pc is not None and old_pc is not pc:
            logger.warning("Replacing existing connection for webrtc_id {}", offer.webrtc_id)
            try:
                await old_pc.close()
            except Exception:
                logger.warning("Error closing old peer connection for {}", offer.webrtc_id)

        # Add video track
        stream_queue = self._frame_broadcaster.register(webrtc_id=offer.webrtc_id)
        track = InferenceVideoStreamTrack(stream_queue=stream_queue)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def connection_state_change() -> None:
            # Only tear down on terminal states. "disconnected" is transient and may
            # recover on its own (e.g. during the inference pause of a model switch),
            # so cleaning it up here would permanently kill a recoverable stream.
            state = pc.connectionState
            if state in ("failed", "closed"):
                logger.debug("Connection state changed to '{}' for webrtc_id {}", state, offer.webrtc_id)
                await self.cleanup_connection(offer.webrtc_id, pc=pc)

        # Set remote description from client's offer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Mangle SDP if public IP is configured
        sdp = pc.localDescription.sdp
        if self._settings.advertise_ip:
            sdp = await self._sdp_handler.mangle_sdp(sdp, self._settings.advertise_ip)

        return Answer(sdp=sdp, type=pc.localDescription.type)

    def set_input(self, data: InputData) -> None:
        """Set input data for specific WebRTC connection"""
        self._input_data[data.webrtc_id] = {
            "conf_threshold": data.conf_threshold,
            "updated_at": time.monotonic(),
        }

    async def cleanup_connection(self, webrtc_id: str, pc: RTCPeerConnection | None = None) -> None:
        """Clean up a specific WebRTC connection by its ID.

        Args:
            webrtc_id: The id of the connection to clean up.
            pc: If provided, the cleanup is only performed when this exact peer connection
                is still the active one for ``webrtc_id``. This guards against a stale
                "closed" handler from a replaced connection tearing down its successor.
        """
        async with self._lock:
            current = self._pcs.get(webrtc_id)
            if pc is not None and current is not pc:
                # The connection was already replaced by a newer one; ignore this stale event.
                logger.debug("Ignoring stale cleanup for webrtc_id {} (connection already replaced)", webrtc_id)
                return
            self._pcs.pop(webrtc_id, None)
            self._frame_broadcaster.unregister(webrtc_id=webrtc_id)
            self._input_data.pop(webrtc_id, None)
            pc_to_close = current

        # Close the peer connection outside the lock to avoid blocking new offers
        if pc_to_close is not None:
            logger.debug("Cleaning up connection: {}", webrtc_id)
            try:
                await pc_to_close.close()
            except Exception:
                logger.debug("Error closing peer connection for {}", webrtc_id)
            logger.debug("Connection {} successfully closed.", webrtc_id)

    async def cleanup(self) -> None:
        """Clean up all connections"""
        async with self._lock:
            for webrtc_id, pc in self._pcs.items():
                await pc.close()
                self._frame_broadcaster.unregister(webrtc_id=webrtc_id)
            self._pcs.clear()
            self._input_data.clear()
            self._frame_broadcaster.cleanup()
