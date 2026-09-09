<!-- markdownlint-disable MD013 MD033 MD041 MD042 -->

<div align="center">

<img src="../assets/geti-header.png" alt="Geti™ - Build and deploy computer vision AI models with minimal effort and data">

**Full-stack web application to build and deploy computer vision AI models, powered by the [getitune](../library) library.**

[![python](https://img.shields.io/badge/python-3.13-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.14-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2026.3-purple)]()

[Application](#geti-application) •
[Docs](#documentation) •
[License](#license)

</div>

## Geti Application

Geti is an end-to-end application for building and deploying computer vision AI models.
It provides an intuitive graphical interface to upload image or video data, annotate datasets,
train and optimize models, and run real-time inference through configurable pipelines.

<p align="center">
  <img src="../assets/application.gif" alt="Application demo" width="100%">
</p>

Main capabilities:

- **No-code model lifecycle**: move from data upload and annotation to training, evaluation, and deployment in one UI.
- **Built-in data and annotation workflows**: manage datasets, labels, and revisions with manual and AI-assisted annotation tools.
- **Pipeline-based deployment**: connect sources (cameras or files) to trained models and route predictions to sinks such as storage, MQTT, or webhooks.
- **Edge-oriented optimization**: export OpenVINO-optimized models for efficient inference on Intel hardware, with support for other accelerators.

### Installation

Geti can be installed as a **Windows app**, run as a **Docker container**, built **from source**, or set up with the
**install script**. Pick the option that best fits your environment.

For complete, step-by-step instructions - including prerequisites, GPU/accelerator support, TLS and TURN configuration,
air-gapped setup, and troubleshooting - see the [Installation guide](./docs/install.md).

## Documentation

Please check the [documentation website](https://docs.geti.intel.com/) for detailed guides, API reference,
and other resources to help you get the most out of Geti.

> **Upgrading an existing installation?** See the [Upgrade guide](./docs/upgrade.md) for how to move to a newer
> version (Docker or Windows MSIX) while preserving your projects, datasets and models, with automatic rollback
> if a migration fails.

<details>
<summary><strong>Advanced: generate the API spec from source </strong></summary>

The OpenAPI specification for the Geti REST API can be generated with the `generate-api-spec` command:

```bash
# From the repo root
cd application/backend

# Generate the OpenAPI spec and save it to a custom location
just gen-api-spec --output-path="openapi.json"
```

</details>

## License

The Geti source code is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). The Windows MSIX App is licensed under the [Intel Simplified Software License](https://software.intel.com/sites/landingpage/pintool/intel-simplified-software-license.txt).
For more information, refer to the [LICENSE](../LICENSE) page.
