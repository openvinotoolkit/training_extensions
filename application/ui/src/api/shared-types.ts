// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { components } from './openapi-spec';

export type Label = components['schemas']['LabelView'];

export type Pipeline = components['schemas']['PipelineView'];

export type Model = components['schemas']['ModelView'];
export type ModelVariant = components['schemas']['ModelVariantView'];
export type ModelArchitecture = components['schemas']['ModelArchitectureView'];
export type ModelArchitectureWithPerformanceCategory = ModelArchitecture & { performanceCategory?: string };
export type BenchmarkMetrics = components['schemas']['BenchmarkMetrics'];
export type ModelFormat = components['schemas']['ModelFormat'];
export type RecommendedModelArchitectures = components['schemas']['TopPicks'];
export type Evaluation = components['schemas']['EvaluationView'];
export type Metric = components['schemas']['MetricView'];
export type LineMetric = components['schemas']['LineMetric'];

export type Job = components['schemas']['JobView'];
export type TrainJob = Job & {
    job_type: 'train';
    metadata: components['schemas']['TrainingMetadata'];
};
export type QuantizeJob = Job & {
    job_type: 'quantize';
    metadata: components['schemas']['QuantizationMetadata'];
};
export type ExportDatasetJob = Job & {
    type: 'export_dataset';
    metadata: components['schemas']['ExportDatasetMetadata'];
};
export type ExportDatasetMetadata = ExportDatasetJob['metadata'];

export type PrepareImportDatasetJob = Job & {
    type: 'prepare_dataset_for_import';
    metadata: components['schemas']['PrepareDatasetForImportRequest'];
};

export type DatasetStatisticsView = components['schemas']['DatasetStatisticsView'];

export type MediaImage = components['schemas']['ImageView'];
export type MediaVideo = components['schemas']['VideoView'];
type MediaVideoFrameDTO = components['schemas']['VideoFrameView'];
export type MediaVideoFrame = Omit<MediaVideo, 'type'> & {
    frame_number: number;
    frame_stride: number;
    type: 'video_frame';
};

export type MediaDTO = MediaImage | MediaVideo | MediaVideoFrameDTO;

export type Media = MediaImage | MediaVideo | MediaVideoFrame;

export type TrainingDevice = components['schemas']['DeviceInfoView'];

export type DatasetSubset = components['schemas']['DatasetItemSubset'];
export type DatasetItem = components['schemas']['DatasetItemView'];
export type DatasetRevision = components['schemas']['DatasetRevisionView'];
export type DatasetRevisionItem = components['schemas']['DatasetRevisionItemView'];

export type Project = components['schemas']['ProjectView'];
export type ProjectCreate = components['schemas']['ProjectCreate'];

export type TaskType = components['schemas']['TaskType'];
export type Task = components['schemas']['TaskView'];

export type ImagesFolderSourceConfig = components['schemas']['ImagesFolderSourceConfigView'];
export type IPCameraSourceConfig = components['schemas']['IPCameraSourceConfigView'];
export type USBCameraSourceConfig = components['schemas']['USBCameraSourceConfigView'];
export type VideoFileSourceConfig = components['schemas']['VideoFileSourceConfigView'];
type DisconnectedSourceConfig = components['schemas']['DisconnectedSourceConfigView'];

export type SourceConfig =
    | DisconnectedSourceConfig
    | USBCameraSourceConfig
    | IPCameraSourceConfig
    | VideoFileSourceConfig
    | ImagesFolderSourceConfig;

export type SourceConfigPayload = Exclude<SourceConfig, DisconnectedSourceConfig>;

export type LocalFolderSinkConfig = components['schemas']['FolderSinkConfigView'];
export type MqttSinkConfig = components['schemas']['MqttSinkConfigView'];
export type WebhookSinkConfig = components['schemas']['WebhookSinkConfigView'];
export type RosSinkConfig = components['schemas']['RosSinkConfigView'];
export type SinkOutputFormats = LocalFolderSinkConfig['output_formats'];

export type SinkConfig = LocalFolderSinkConfig | MqttSinkConfig | WebhookSinkConfig | RosSinkConfig;

export type AnnotationDTO = components['schemas']['DatasetItemAnnotation'];
export type PredictionDTO = components['schemas']['DatasetItemAnnotation'];
export type DatasetItemAnnotationStatus = components['schemas']['DatasetItemAnnotationStatus'];
export type FilterByStatusKey = 'all' | DatasetItemAnnotationStatus;

export type AnnotatedVideoFrame = components['schemas']['AnnotatedVideoFrame'];
export type VideoFramePrediction = {
    media: components['schemas']['BatchInferenceMedia'];
    prediction: components['schemas']['DatasetItemAnnotation'][];
};

export type PredictionVideoRangePayload = components['schemas']['VideoRange'];

export type AnnotationType = components['schemas']['AnnotationType'];

export type Point = components['schemas']['Point'];
export type Rect = components['schemas']['Rectangle'];
export type Polygon = components['schemas']['Polygon'];
export type FullImage = components['schemas']['FullImage'];

export type Shape = Rect | Polygon | FullImage;

export type BoolConfigurableParameter = components['schemas']['BoolParameterView'];
export type StringConfigurableParameter = components['schemas']['StringParameterView'];
type IntConfigurableParameter = components['schemas']['IntParameterView'];
type FloatConfigurableParameter = components['schemas']['FloatParameterView'];
export type FloatConfigurableRangeParameter = components['schemas']['FloatRangeParameterView'];

export type NumberConfigurableParameter = IntConfigurableParameter | FloatConfigurableParameter;

type CreateEnumerableConfigurableParameterType<T extends StringConfigurableParameter | NumberConfigurableParameter> =
    Omit<T, 'allowed_values' | 'value' | 'default_value'> & {
        allowed_values: Exclude<T['allowed_values'], null | undefined>;
        value: Exclude<T['value'], null | undefined>;
        default_value: Exclude<T['default_value'], null | undefined>;
    };

export type NumberEnumConfigurableParameter = CreateEnumerableConfigurableParameterType<NumberConfigurableParameter>;
export type StringEnumConfigurableParameter = CreateEnumerableConfigurableParameterType<StringConfigurableParameter>;

export type ConfigurableParameter =
    | BoolConfigurableParameter
    | StringConfigurableParameter
    | IntConfigurableParameter
    | FloatConfigurableParameter
    | FloatConfigurableRangeParameter;

export type ConfigurableParameterGroup = components['schemas']['ConfigurableParameterGroupView'];
export type TrainingConfigurationParameter = ConfigurableParameter | ConfigurableParameterGroup;

export type TrainingConfiguration = components['schemas']['TrainingConfigurationView'];

export type TrainingRequestPayload = components['schemas']['TrainingRequest'];
export type TrainingConfigurationRequestPayload = {
    [key: string]: unknown;
};

export type Pagination = components['schemas']['Pagination'];
export type MediaWithPagination = components['schemas']['MediaWithPagination'];
export type DatasetFormat = components['schemas']['DatasetFormat'];
export type DeviceInfo = components['schemas']['DeviceInfoView'];
export type MediaListPredictionRequest = components['schemas']['MediaListPredictionRequest'];

export type PipelineHealth = components['schemas']['PipelineHealth'];
export type PipelineComponentsHealth = components['schemas']['PipelineComponentsHealth'];
export type PipelineStatus = components['schemas']['Status'];

export type DatasetView = components['schemas']['DatasetViewView'];
