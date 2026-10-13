# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, computed_field

from app.core.models import BaseIDNameModel
from app.models import (
    DisconnectedSinkConfig,
    FolderSinkConfig,
    MqttSinkConfig,
    OutputFormat,
    RosSinkConfig,
    SinkType,
    WebhookSinkConfig,
)
from app.models.sink import (
    DisconnectedConfig,
    FolderConfig,
    HttpHeaders,
    HttpMethod,
    MqttConfig,
    RosConfig,
    WebhookConfig,
)


class DisconnectedSinkConfigView(DisconnectedSinkConfig):
    config_data: DisconnectedConfig = Field(exclude=True)
    model_config = {
        "json_schema_extra": {
            "example": {
                "sink_type": "disconnected",
                "name": "No Sink",
                "id": "00000000-0000-0000-0000-000000000000",
            }
        }
    }


class FolderSinkConfigView(FolderSinkConfig):
    config_data: FolderConfig = Field(exclude=True)

    @computed_field
    @property
    def folder_path(self) -> str:
        return self.config_data.folder_path

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "b5787c06-964b-4097-8eca-238b8cf79fc8",
                "sink_type": "folder",
                "name": "Local Folder",
                "output_formats": ["image_original", "image_with_predictions", "predictions"],
                "rate_limit": 0.2,
                "folder_path": "/path/to/output",
            }
        }
    }


class MqttSinkConfigView(MqttSinkConfig):
    config_data: MqttConfig = Field(exclude=True)

    @computed_field
    @property
    def broker_host(self) -> str:
        return self.config_data.broker_host

    @computed_field
    @property
    def broker_port(self) -> int:
        return self.config_data.broker_port

    @computed_field
    @property
    def topic(self) -> str:
        return self.config_data.topic

    @computed_field
    @property
    def auth_required(self) -> bool:
        return self.config_data.auth_required

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "c1a70159-9c9e-4f02-821a-02576321056c",
                "sink_type": "mqtt",
                "name": "Local MQTT Broker",
                "broker_host": "localhost",
                "broker_port": 1883,
                "topic": "predictions",
                "auth_required": True,
                "output_formats": ["predictions"],
            }
        }
    }


class RosSinkConfigView(RosSinkConfig):
    config_data: RosConfig = Field(exclude=True)

    @computed_field
    @property
    def topic(self) -> str:
        return self.config_data.topic

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "6f1d96ac-db38-42a9-9a11-142d404f493f",
                "sink_type": "ros",
                "name": "ROS2 Predictions Topic",
                "output_formats": ["predictions"],
                "topic": "/predictions",
            }
        }
    }


class WebhookSinkConfigView(WebhookSinkConfig):
    config_data: WebhookConfig = Field(exclude=True)

    @computed_field
    @property
    def webhook_url(self) -> str:
        return self.config_data.webhook_url

    @computed_field
    @property
    def http_method(self) -> HttpMethod:
        return self.config_data.http_method

    @computed_field
    @property
    def headers(self) -> HttpHeaders | None:
        return self.config_data.headers

    @computed_field
    @property
    def timeout(self) -> int:
        return self.config_data.timeout

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "39ba53e5-9a03-44fc-b78a-83245cf14676",
                "sink_type": "webhook",
                "name": "Webhook Endpoint",
                "output_formats": ["predictions"],
                "webhook_url": "https://example.com/webhook",
                "http_method": "PUT",
                "headers": {"Authorization": "Bearer YOUR_TOKEN"},
                "timeout": 10,
            }
        }
    }


SinkView = Annotated[
    DisconnectedSinkConfigView | FolderSinkConfigView | MqttSinkConfigView | RosSinkConfigView | WebhookSinkConfigView,
    Field(discriminator="sink_type"),
]

SinkViewAdapter: TypeAdapter[SinkView] = TypeAdapter(SinkView)


class BaseSinkConfigCreate(BaseIDNameModel):
    rate_limit: float | None = Field(default=None, gt=0)
    output_formats: list[OutputFormat]


class FolderSinkConfigCreate(BaseSinkConfigCreate):
    sink_type: Literal[SinkType.FOLDER]
    folder_path: str

    @property
    def config_data(self) -> FolderConfig:
        return FolderConfig(folder_path=self.folder_path)


class MqttSinkConfigCreate(BaseSinkConfigCreate):
    sink_type: Literal[SinkType.MQTT]
    broker_host: str
    broker_port: int
    topic: str
    auth_required: bool

    @property
    def config_data(self) -> MqttConfig:
        return MqttConfig(
            broker_host=self.broker_host,
            broker_port=self.broker_port,
            topic=self.topic,
            auth_required=self.auth_required,
        )


class RosSinkConfigCreate(BaseSinkConfigCreate):
    sink_type: Literal[SinkType.ROS]
    topic: str

    @property
    def config_data(self) -> RosConfig:
        return RosConfig(topic=self.topic)


class WebhookSinkConfigCreate(BaseSinkConfigCreate):
    sink_type: Literal[SinkType.WEBHOOK]
    webhook_url: str
    http_method: HttpMethod
    headers: HttpHeaders | None
    timeout: int = Field(gt=0, strict=True)

    @property
    def config_data(self) -> WebhookConfig:
        return WebhookConfig(
            webhook_url=self.webhook_url, http_method=self.http_method, headers=self.headers, timeout=self.timeout
        )


SinkCreate = Annotated[
    FolderSinkConfigCreate | MqttSinkConfigCreate | RosSinkConfigCreate | WebhookSinkConfigCreate,
    Field(discriminator="sink_type"),
]


SinkCreateAdapter: TypeAdapter[SinkCreate] = TypeAdapter(SinkCreate)
