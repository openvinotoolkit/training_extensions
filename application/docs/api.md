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

| Method   | Path                                              | Payload | Return                         | Description                                |
| -------- | ------------------------------------------------- | ------- | ------------------------------ | ------------------------------------------ |
| `GET`    | `/api/projects/<id>/dataset/media`                | -       | list of dataset media          | List the dataset media (images and videos) |
| `GET`    | `/api/projects/<id>/dataset/media/<id>`           | -       | dataset media info             | Get info about a dataset media             |
| `GET`    | `/api/projects/<id>/dataset/media/<id>/frames`    | -       | list of annotated video frames | List the annotated video frames            |
| `GET`    | `/api/projects/<id>/dataset/media/<id>/binary`    | -       | binary                         | Get the image data of a media (full res)   |
| `GET`    | `/api/projects/<id>/dataset/media/<id>/thumbnail` | -       | binary                         | Get the thumbnail of a media               |
| `POST`   | `/api/projects/<id>/dataset/media`                | binary  | media info                     | Upload an image or a video to the dataset  |
| `DELETE` | `/api/projects/<id>/dataset/media/<id>`           | -       | -                              | Delete a dataset media                     |

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

| Method   | Path                            | Payload | Return           | Description                            |
| -------- | ------------------------------- | ------- | ---------------- | -------------------------------------- |
| `GET`    | `/api/staged_datasets`          | -       | list of datasets | List datasets from the staging area    |
| `POST`   | `/api/staged_datasets`          | binary  | item info        | Upload dataset archive to staging area |
| `GET`    | `/api/staged_datasets/<id>`     | -       | item info        | Get info about staged dataset          |
| `GET`    | `/api/staged_datasets/<id>/zip` | -       | binary           | Download archive from the staging area |
| `DELETE` | `/api/staged_datasets/<id>`     | -       | -                | Remove dataset from the staging area   |

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
