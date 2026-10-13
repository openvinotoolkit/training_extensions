# REST API

## Input / output

### Sources

| Method   | Path                       | Payload       | Return          | Description                       |
| -------- | -------------------------- | ------------- | --------------- | --------------------------------- |
| `POST`   | `/api/sources`             | source config | source id       | Create and configure a new source |
| `GET`    | `/api/sources`             | -             | list of sources | List the available sources        |
| `GET`    | `/api/sources/<id>`        | -             | source info     | Get info about a source           |
| `PATCH`  | `/api/sources/<id>`        | source config | -               | Reconfigure an existing source    |
| `POST`   | `/api/sources/<id>:export` | -             | yaml file       | Export a source to file           |
| `POST`   | `/api/sources:import`      | yaml file     | source id       | Import a source from file         |
| `DELETE` | `/api/sources/<id>`        | -             | -               | Remove a source                   |

| Method | Path                             | Payload       | Return     | Description                                                   |
| ------ | -------------------------------- | ------------- | ---------- | ------------------------------------------------------------- |
| `POST` | `/api/sources/media`             | binary        | video path | Upload a video file to use as a video file source             |
| `POST` | `/api/sources/media:from-upload` | `{upload_id}` | video path | Same, from a completed [resumable upload](#resumable-uploads) |

### Sinks

| Method   | Path                     | Payload     | Return        | Description                     |
| -------- | ------------------------ | ----------- | ------------- | ------------------------------- |
| `POST`   | `/api/sinks`             | sink config | sink id       | Create and configure a new sink |
| `GET`    | `/api/sinks`             | -           | list of sinks | List the available sinks        |
| `GET`    | `/api/sinks/<id>`        | -           | sink info     | Get info about a sink           |
| `PATCH`  | `/api/sinks/<id>`        | sink config | -             | Reconfigure an existing sink    |
| `POST`   | `/api/sinks/<id>:export` | -           | yaml file     | Export a sink to file           |
| `POST`   | `/api/sinks:import`      | yaml file   | sink id       | Import a sink from file         |
| `DELETE` | `/api/sinks/<id>`        | -           | -             | Remove a sink                   |

## Projects

| Method   | Path                                        | Payload            | Return           | Description                       |
| -------- | ------------------------------------------- | ------------------ | ---------------- | --------------------------------- |
| `POST`   | `/api/projects`                             | name, task, labels | project info     | Create a new project              |
| `GET`    | `/api/projects`                             | -                  | list of projects | List the available projects       |
| `GET`    | `/api/projects/<id>`                        | -                  | project info     | Get info about a project          |
| `PATCH`  | `/api/projects/<id>`                        | name               | project info     | Rename a project                  |
| `DELETE` | `/api/projects/<id>`                        | -                  | -                | Delete a project                  |
| `PATCH`  | `/api/projects/<id>/labels`                 | labels to change   | task and labels  | Add, remove or edit labels        |
| `GET`    | `/api/projects/<id>/training_configuration` | -                  | training config  | Get the training configuration    |
| `PATCH`  | `/api/projects/<id>/training_configuration` | training config    | -                | Update the training configuration |

### Pipelines

| Method  | Path                                  | Payload                    | Return        | Description                           |
| ------- | ------------------------------------- | -------------------------- | ------------- | ------------------------------------- |
| `GET`   | `/api/projects/<id>/pipeline`         | -                          | pipeline info | Get info about a project's pipeline   |
| `PATCH` | `/api/projects/<id>/pipeline`         | ids of source, sink, model | pipeline info | Reconfigure the project's pipeline    |
| `POST`  | `/api/projects/<id>/pipeline:enable`  | -                          | pipeline info | Activate a project's pipeline         |
| `POST`  | `/api/projects/<id>/pipeline:disable` | -                          | pipeline info | Deactivate a project's pipeline       |
| `POST`  | `/api/projects/<id>/pipeline:capture` | -                          | -             | Collect the next frame to the dataset |

#### Inference metrics

| Method | Path                                  | Payload | Return       | Description                                      |
| ------ | ------------------------------------- | ------- | ------------ | ------------------------------------------------ |
| `GET`  | `/api/projects/<id>/pipeline/metrics` | -       | metrics info | Get inference metrics (latency, throughput, ...) |

## Media

| Method   | Path                                              | Payload              | Return                         | Description                                                   |
| -------- | ------------------------------------------------- | -------------------- | ------------------------------ | ------------------------------------------------------------- |
| `GET`    | `/api/projects/<id>/dataset/media`                | -                    | list of dataset media          | List the dataset media (images and videos)                    |
| `GET`    | `/api/projects/<id>/dataset/media/<id>`           | -                    | dataset media info             | Get info about a dataset media                                |
| `GET`    | `/api/projects/<id>/dataset/media/<id>/frames`    | -                    | list of annotated video frames | List the annotated video frames                               |
| `GET`    | `/api/projects/<id>/dataset/media/<id>/binary`    | -                    | binary                         | Get the image data of a media (full res)                      |
| `GET`    | `/api/projects/<id>/dataset/media/<id>/thumbnail` | -                    | binary                         | Get the thumbnail of a media                                  |
| `POST`   | `/api/projects/<id>/dataset/media`                | binary               | media info                     | Upload an image or a video to the dataset                     |
| `POST`   | `/api/projects/<id>/dataset/media:from-upload`    | `{upload_id, name?}` | media info                     | Same, from a completed [resumable upload](#resumable-uploads) |
| `DELETE` | `/api/projects/<id>/dataset/media/<id>`           | -                    | -                              | Delete a dataset media                                        |

> `GET /api/projects/<id>/dataset/media` accepts an optional `dataset_view_id` query parameter: when provided,
> only the media assigned to that [dataset view](#views) are returned. This is the only endpoint to list the
> content of a dataset view.

### Predictions

| Method | Path                                       | Payload              | Return                 | Description                                 |
| ------ | ------------------------------------------ | -------------------- | ---------------------- | ------------------------------------------- |
| `POST` | `/api/projects/<id>/dataset/media:predict` | model id, media list | batch inference result | Get predictions for one or more media items |

> **Deprecated:** `POST /api/projects/<id>/dataset/media/media:predict` (with the duplicated `media` path
> segment) is deprecated in favor of `POST /api/projects/<id>/dataset/media:predict` above. The deprecated
> path is marked `deprecated: true` in the OpenAPI spec and will be removed in version 3.4.

### Annotations

| Method   | Path                                                | Payload         | Return          | Description                               |
| -------- | --------------------------------------------------- | --------------- | --------------- | ----------------------------------------- |
| `GET`    | `/api/projects/<id>/dataset/media/<id>/annotations` | -               | annotation info | Get the annotation/prediction for a media |
| `POST`   | `/api/projects/<id>/dataset/media/<id>/annotations` | annotation info | annotation info | Annotate a media                          |
| `DELETE` | `/api/projects/<id>/dataset/media/<id>/annotations` | -               | -               | Delete the annotation for a media         |

## Dataset items

| Method | Path                                    | Payload | Return                | Description                                        |
| ------ | --------------------------------------- | ------- | --------------------- | -------------------------------------------------- |
| `GET`  | `/api/projects/<id>/dataset/items`      | -       | list of dataset items | List the dataset items (option 'with_annotations') |
| `GET`  | `/api/projects/<id>/dataset/items/<id>` | -       | dataset item info     | Get info about a dataset item                      |

### Tags

| Method  | Path                                         | Payload                   | Return       | Description                                 |
| ------- | -------------------------------------------- | ------------------------- | ------------ | ------------------------------------------- |
| `GET`   | `/api/projects/<id>/dataset/items/<id>/tags` | -                         | list of tags | List the tags of a dataset item             |
| `GET`   | `/api/projects/<id>/dataset/tags`            | -                         | list of tags | List the tags used in the dataset           |
| `PATCH` | `/api/projects/<id>/dataset/items/tags`      | items, tags to add/remove | -            | Apply or remove tags from one or more items |

### Views

| Method   | Path                                          | Payload   | Return        | Description                        |
| -------- | --------------------------------------------- | --------- | ------------- | ---------------------------------- |
| `POST`   | `/api/projects/<id>/dataset/views`            | name      | view info     | Create a new dataset view          |
| `GET`    | `/api/projects/<id>/dataset/views`            | -         | list of views | List the dataset views             |
| `GET`    | `/api/projects/<id>/dataset/views/<id>`       | -         | view info     | Get info about a dataset view      |
| `PATCH`  | `/api/projects/<id>/dataset/views/<id>`       | name      | view info     | Rename a dataset view              |
| `POST`   | `/api/projects/<id>/dataset/views/<id>/media` | media ids | -             | Assign media to a dataset view     |
| `DELETE` | `/api/projects/<id>/dataset/views/<id>/media` | media ids | -             | Unassign media from a dataset view |
| `DELETE` | `/api/projects/<id>/dataset/views/<id>`       | -         | -             | Delete a dataset view              |

> **Listing the content of a view.** There is no endpoint to list the content of a view in this section: the
> existing dataset endpoints accept an optional `dataset_view_id` query parameter which restricts the results to
> the media/items assigned to that view, with the same filtering, sorting and pagination options:
>
> | Method | Path                                                              | Description                         |
> | ------ | ----------------------------------------------------------------- | ----------------------------------- |
> | `GET`  | `/api/projects/<id>/dataset/media?dataset_view_id=<view_id>`      | List the media assigned to the view |
> | `GET`  | `/api/projects/<id>/dataset/items?dataset_view_id=<view_id>`      | List the dataset items of the view  |
> | `GET`  | `/api/projects/<id>/dataset/statistics?dataset_view_id=<view_id>` | Get the statistics of the view      |

### Models

| Method   | Path                                                                      | Payload | Return           | Description                                                             |
| -------- | ------------------------------------------------------------------------- | ------- | ---------------- | ----------------------------------------------------------------------- |
| `GET`    | `/api/projects/<id>/models`                                               | -       | list of models   | List all the models in a project                                        |
| `GET`    | `/api/projects/<id>/models/<model_id>`                                    | -       | model info       | Get info about a specific model                                         |
| `GET`    | `/api/projects/<id>/models/<model_id>/labels`                             | -       | labels           | Get the labels used to train the model                                  |
| `GET`    | `/api/projects/<id>/models/<model_id>/variants/<model_variant_id>/binary` | -       | zip              | Download model binary of the requested model variant                    |
| `DELETE` | `/api/projects/<id>/models/<model_id>`                                    | -       | -                | Delete a model (option 'weights_only')                                  |
| `GET`    | `/api/projects/<id>/models/<model_id>/training_metrics`                   | -       | training metrics | Get training metrics                                                    |
| `GET`    | `/api/projects/<id>/models/<model_id>/logs`                               | -       | training log     | Get training logs (supports Accept: text/plain or application/x-ndjson) |

### Dataset revisions (training datasets, etc...)

| Method   | Path                                                        | Payload | Return        | Description                                        |
| -------- | ----------------------------------------------------------- | ------- | ------------- | -------------------------------------------------- |
| `GET`    | `/api/projects/<id>/dataset_revisions/items`                | -       | list of items | List the dataset items (option 'with_annotations') |
| `GET`    | `/api/projects/<id>/dataset_revisions/items/<id>`           | -       | item info     | Get info about a dataset item                      |
| `GET`    | `/api/projects/<id>/dataset_revisions/items/<id>/binary`    | -       | binary        | Get the image data of a dataset item (full res)    |
| `GET`    | `/api/projects/<id>/dataset_revisions/items/<id>/thumbnail` | -       | binary        | Get the thumbnail of a dataset item                |
| `DELETE` | `/api/projects/<id>/dataset_revisions`                      | -       | -             | Remove the dataset files to free space             |

### Staged datasets

| Method   | Path                               | Payload       | Return           | Description                                                   |
| -------- | ---------------------------------- | ------------- | ---------------- | ------------------------------------------------------------- |
| `GET`    | `/api/staged_datasets`             | -             | list of datasets | List datasets from the staging area                           |
| `POST`   | `/api/staged_datasets`             | binary        | item info        | Upload dataset archive to staging area                        |
| `POST`   | `/api/staged_datasets:from-upload` | `{upload_id}` | item info        | Same, from a completed [resumable upload](#resumable-uploads) |
| `GET`    | `/api/staged_datasets/<id>`        | -             | item info        | Get info about staged dataset                                 |
| `GET`    | `/api/staged_datasets/<id>/zip`    | -             | binary           | Download archive from the staging area                        |
| `DELETE` | `/api/staged_datasets/<id>`        | -             | -                | Remove dataset from the staging area                          |

### Resumable uploads

Large files (videos, dataset archives) can be uploaded in chunks with the [TUS 1.0.0](https://tus.io/protocols/resumable-upload)
protocol, so that an interrupted transfer resumes from the last byte the server committed instead of restarting, and
the client can report progress. Uploading is a two-step process:

1. Transfer the file to `/api/uploads` with any TUS client (the UI uses [`tus-js-client`](https://github.com/tus/tus-js-client)).
2. Hand the completed upload to one of the `:from-upload` endpoints, which move the file to its final location
   (no copy) and return the same response as their multipart twin:
   - `POST /api/projects/<id>/dataset/media:from-upload` (image or video, optional `name` override)
   - `POST /api/staged_datasets:from-upload` (`.zip` dataset archive)
   - `POST /api/sources/media:from-upload` (video file source)

The multipart endpoints (`POST /api/projects/<id>/dataset/media`, `POST /api/staged_datasets`,
`POST /api/sources/media`) remain available for direct uploads.

| Method    | Path                | TUS extension | Description                                                                                                                                                               |
| --------- | ------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPTIONS` | `/api/uploads`      | core          | Advertise `Tus-Version`, `Tus-Extension`, `Tus-Max-Size` and `Tus-Checksum-Algorithm`                                                                                     |
| `POST`    | `/api/uploads`      | creation      | Create an upload (`Upload-Length` or `Upload-Defer-Length: 1`, `Upload-Metadata`); the body may carry the first chunk. Returns `201` with `Location` and `Upload-Expires` |
| `HEAD`    | `/api/uploads/<id>` | core          | Return `204` with the committed `Upload-Offset` (and `Upload-Length`) to resume from                                                                                      |
| `PATCH`   | `/api/uploads/<id>` | core          | Append a chunk (`Content-Type: application/offset+octet-stream`) at `Upload-Offset`; returns `204` with the new offset                                                    |
| `DELETE`  | `/api/uploads/<id>` | termination   | Cancel the upload and discard the bytes received so far                                                                                                                   |
| `GET`     | `/api/uploads/<id>` | -             | Upload state as JSON (`UploadView`)                                                                                                                                       |
| `GET`     | `/api/uploads`      | -             | List uploads, most recent first                                                                                                                                           |

Every TUS request must send `Tus-Resumable: 1.0.0`, otherwise the server answers `412`. `Upload-Metadata` values are
base64-encoded; the server reads `filename` (sanitized, never used as a file path) and `filetype`. The file extension is
validated when the upload is consumed, not when it is created.

An upload goes through the states `pending` → `in_progress` → `completed` → `consumed`:

```json
{
  "id": "0f8c6c55-8f2c-4c1b-9a8e-3c2f0e1f4b6a",
  "filename": "grapes.mp4",
  "size": 734003200,
  "offset": 412876800,
  "state": "in_progress",
  "created_at": "2026-09-30T12:00:00Z",
  "expires_at": "2026-10-01T12:00:00Z"
}
```

Error responses:

| Status | Meaning                                                                                                                                |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `400`  | Malformed headers (e.g. invalid `Upload-Metadata` base64), or `X-HTTP-Method-Override` was sent (not supported)                        |
| `404`  | Unknown upload                                                                                                                         |
| `409`  | `Upload-Offset` does not match the committed offset (use `HEAD` to resynchronize), or the upload is not complete yet when consuming it |
| `410`  | The upload expired or was already consumed                                                                                             |
| `413`  | `Upload-Length`, or the bytes sent, exceed the upload length or `Tus-Max-Size`                                                         |
| `415`  | `PATCH` without `Content-Type: application/offset+octet-stream`                                                                        |
| `423`  | Another `PATCH` is currently writing to the same upload                                                                                |
| `460`  | The chunk does not match its `Upload-Checksum` (`sha1`, `sha256` or `md5`); the chunk is discarded                                     |

An upload can be consumed only once. If the consuming endpoint rejects the file (e.g. unsupported extension), the upload
stays `completed`, so the client can retry or `DELETE` it.

The bytes are stored as `<UPLOADS_DIR>/<upload_id>.part` and the metadata in the `uploads` database table, which is the
source of truth for the offset: after a crash, bytes beyond the committed offset are truncated on the next `HEAD` or
`PATCH`. The following settings apply:

| Environment variable | Default                | Description                                                                          |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------ |
| `UPLOADS_DIR`        | `<DATA_DIR>/uploads`   | Directory holding the partial uploads                                                |
| `UPLOAD_MAX_SIZE`    | `53687091200` (50 GiB) | Maximum upload size in bytes, advertised as `Tus-Max-Size`                           |
| `UPLOAD_TTL_HOURS`   | `24`                   | Inactivity period after which an unfinished upload expires (each `PATCH` extends it) |

A background task removes expired and consumed uploads every 15 minutes. Concurrent `PATCH` requests are serialized
within the backend process, so do not share one uploads directory between several backend processes.

## Jobs

| Method | Path                    | Payload             | Return       | Description                                        |
| ------ | ----------------------- | ------------------- | ------------ | -------------------------------------------------- |
| `POST` | `/api/jobs`             | job type and params | job id       | Create and submit a new job                        |
| `GET`  | `/api/jobs`             | -                   | list of jobs | List the jobs in a project (scheduled or running)  |
| `GET`  | `/api/jobs/<id>`        | -                   | job info     | Get info about a specific job                      |
| `POST` | `/api/jobs/<id>:cancel` | -                   | -            | Cancel a job                                       |
| `GET`  | `/api/jobs/<id>/status` | -                   | job status   | Stream real-time status updates for a specific job |
| `GET`  | `/api/jobs/<id>/logs`   | -                   | job logs     | Stream real-time log output for a specific job     |

Job types:

- `train`
- `quantize`
- `prepare_dataset_for_import`
- `import_dataset_to_existing_project`
- `import_dataset_as_new_project`
- `export_dataset`
- `stage_dataset`

> A `train` job accepts either `dataset_revision_id` (train on an existing revision) or `dataset_view_id`
> (train on the media assigned to a [dataset view](#views)) in its parameters, but not both; when neither is
> given, the job trains on the entire dataset in its most recent state.
