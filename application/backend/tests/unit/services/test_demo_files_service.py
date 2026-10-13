# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from importlib import resources
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.models.model_manifest import License
from app.models.model_revision import ModelFormat
from app.services.demo_files_service import _LICENSE_FILES_BY_NAME, DemoFilesService, load_license_files
from app.services.model_manifest_service import ModelManifestService
from app.supported_models.attributions import ModelAttribution

# Licenses which explicitly allow to satisfy their terms with a link to the license text
# (e.g. CC BY-NC 4.0, Section 3(a)(1)(C)), so no copy is bundled with the exported model.
_LINK_ONLY_LICENSES = {"CC BY-NC 4.0", "CC BY-NC-SA 4.0"}


def _all_used_license_names() -> set[str]:
    """Names of all the licenses used by the static model manifests and by the timm catalog."""
    names = {manifest.license.name for manifest in ModelManifestService.get_model_manifests().values()}
    snapshot = json.loads(
        resources.files("app.supported_models").joinpath("timm_catalog_snapshot.json").read_text(encoding="utf-8")
    )
    names.update(entry["license"] for entry in snapshot["backbones"])
    return names


@pytest.fixture
def fxt_demo_files_service() -> DemoFilesService:
    # A media service without any media: the bundle is built without a sample image.
    media_service = MagicMock()
    media_service.list_media.return_value = []
    return DemoFilesService(media_service=media_service)


class TestDemoFilesServiceLicensing:
    @pytest.mark.parametrize(
        "license_name, expected_files",
        [
            ("AGPL-3.0", {"LICENSE": "GNU AFFERO GENERAL PUBLIC LICENSE"}),
            ("Apache 2.0", {"LICENSE": "Apache License"}),
            ("Apple ML MobileOne License", {"LICENSE": "ML-MobileOne"}),
            ("Apple-AMLR", {"LICENSE": "Apple Machine Learning Research Model"}),
            ("Apple-ASCL", {"LICENSE": "This Apple software is supplied to you by Apple"}),
            ("BSD-3-Clause", {"LICENSE": "Redistribution and use in source and binary forms"}),
            ("DEIMv2 License", {"LICENSE": "# DEIMv2 License"}),
            ("DINOv3 License", {"LICENSE": "# DINOv3 License"}),
            ("EdgeCrafter", {"LICENSE": "# EdgeCrafter License"}),
            ("Fair Noncommercial Research License", {"LICENSE": "# FAIR Noncommercial Research License"}),
            (
                "GEMMA",
                {
                    "LICENSE": "# Gemma Terms of Use",
                    "NOTICE": "Gemma is provided under and subject to the Gemma Terms of Use found at "
                    "ai.google.dev/gemma/terms",
                },
            ),
            ("MIT", {"LICENSE": "MIT License"}),
            ("CC BY-NC 4.0", {}),
            ("CC BY-NC-SA 4.0", {}),
        ],
    )
    @pytest.mark.parametrize("model_format", [ModelFormat.OPENVINO, ModelFormat.ONNX])
    def test_license_referenced_and_bundled_when_required(
        self,
        fxt_demo_files_service: DemoFilesService,
        model_format: ModelFormat,
        license_name: str,
        expected_files: dict[str, str],
    ) -> None:
        license_url = "https://example.com/license"

        files = fxt_demo_files_service.build_demo_files(
            project_id=uuid4(), model_format=model_format, license=License(name=license_name, url=license_url)
        )

        by_name = {f.name: f.data for f in files}
        readme = by_name["README.md"].decode("utf-8")

        # The README always references the license by name and URL.
        assert "## Licensing" in readme
        assert (
            f'This model is distributed under the "{license_name}" license, '
            f"the full text is available [here]({license_url})." in readme
        )

        # Only the required license files are bundled, with the expected content.
        bundled_license_files = {name for name in by_name if name in ("LICENSE", "NOTICE")}
        assert bundled_license_files == set(expected_files)
        for filename, snippet in expected_files.items():
            assert snippet in by_name[filename].decode("utf-8")

        if not expected_files:
            assert "A copy of the license is included" not in readme
        elif len(expected_files) == 1:
            assert "A copy of the license is included in the `LICENSE` file of this archive." in readme
        else:
            assert "A copy of the license is included in the `LICENSE` and `NOTICE` files of this archive." in readme

    def test_no_license_section_without_license(self, fxt_demo_files_service: DemoFilesService) -> None:
        files = fxt_demo_files_service.build_demo_files(project_id=uuid4(), model_format=ModelFormat.OPENVINO)

        by_name = {f.name: f.data for f in files}
        assert "LICENSE" not in by_name
        assert "## Licensing" not in by_name["README.md"].decode("utf-8")

    def test_no_files_for_non_deployable_format(self, fxt_demo_files_service: DemoFilesService) -> None:
        files = fxt_demo_files_service.build_demo_files(
            project_id=uuid4(),
            model_format=ModelFormat.PYTORCH,
            license=License(name="AGPL-3.0", url="https://www.ultralytics.com/legal/agpl-3-0-software-license"),
        )

        assert files == []


class TestLicenseFiles:
    @pytest.mark.parametrize("license_name", sorted(_LICENSE_FILES_BY_NAME))
    def test_license_file_resources_exist(self, license_name: str) -> None:
        license_files = load_license_files(license_name)

        assert "LICENSE" in license_files
        for data in license_files.values():
            assert len(data.strip()) > 0

    def test_unknown_license_has_no_files(self) -> None:
        assert load_license_files("Some Unknown License") == {}

    def test_bundled_license_names_are_used(self) -> None:
        """Guard against a license being renamed in the manifests or in the timm catalog, which would
        silently stop bundling its text with exported models."""
        unused = set(_LICENSE_FILES_BY_NAME) - _all_used_license_names()

        assert not unused, f"License files not referenced by any model manifest: {sorted(unused)}"

    def test_all_used_licenses_are_handled(self) -> None:
        """Every license used by a model must either have its files bundled or be explicitly link-only.

        When adding a model with a new license, check whether its terms require a copy of the license
        (or other notices) to be redistributed with the model, and update `_LICENSE_FILES_BY_NAME` in
        `app/services/demo_files_service.py` or `_LINK_ONLY_LICENSES` in this test accordingly.
        """
        unhandled = _all_used_license_names() - set(_LICENSE_FILES_BY_NAME) - _LINK_ONLY_LICENSES

        assert not unhandled, f"Licenses without redistribution policy: {sorted(unhandled)}"


_MIT_ATTRIBUTION = ModelAttribution(
    name="Foo Net",
    models=("image-classification-timm-foo_*",),
    creators="Foo Research Lab",
    copyright="Copyright (c) 2021 Foo Research Lab\nCopyright (c) Bar Inc.",
    source="https://github.com/foo/foo-net",
)


class TestDemoFilesServiceAttribution:
    @pytest.mark.parametrize("license_name", ["MIT", "BSD-3-Clause"])
    def test_upstream_copyright_notice_in_license_file(
        self, fxt_demo_files_service: DemoFilesService, license_name: str
    ) -> None:
        """MIT and BSD-3-Clause require the original copyright notice to be retained in copies."""
        files = fxt_demo_files_service.build_demo_files(
            project_id=uuid4(),
            model_format=ModelFormat.OPENVINO,
            license=License(name=license_name, url="https://example.com/license"),
            attribution=_MIT_ATTRIBUTION,
        )

        license_text = {f.name: f.data for f in files}["LICENSE"].decode("utf-8")
        assert "Copyright (c) 2021 Foo Research Lab\nCopyright (c) Bar Inc." in license_text
        assert "<copyright notice>" not in license_text

    @pytest.mark.parametrize(
        "license_name, expect_license_file",
        [("MIT", True), ("BSD-3-Clause", True), ("CC BY-NC 4.0", False), ("CC BY-NC-SA 4.0", False)],
    )
    @pytest.mark.parametrize(
        "model_format, format_name", [(ModelFormat.OPENVINO, "OpenVINO IR"), (ModelFormat.ONNX, "ONNX")]
    )
    def test_attribution_and_modifications_in_readme(
        self,
        fxt_demo_files_service: DemoFilesService,
        license_name: str,
        expect_license_file: bool,
        model_format: ModelFormat,
        format_name: str,
    ) -> None:
        license_url = "https://example.com/license"

        files = fxt_demo_files_service.build_demo_files(
            project_id=uuid4(),
            model_format=model_format,
            license=License(name=license_name, url=license_url),
            attribution=_MIT_ATTRIBUTION,
        )

        by_name = {f.name: f.data for f in files}
        readme = by_name["README.md"].decode("utf-8")
        assert ("LICENSE" in by_name) is expect_license_file
        # Link to the license text
        assert f"[here]({license_url})" in readme
        # Identification of the creators, source and copyright notice of the original work
        assert "### Attribution" in readme
        assert (
            'This model is derived from the pretrained model "Foo Net", created by Foo Research Lab, '
            "available at <https://github.com/foo/foo-net>." in readme
        )
        assert "```text\nCopyright (c) 2021 Foo Research Lab\nCopyright (c) Bar Inc.\n```" in readme
        # Indication of the modifications and reference to the disclaimer of warranties
        assert "### Modifications" in readme
        assert f"fine-tuned with Geti on a custom dataset and exported to the {format_name} format" in readme
        assert "disclaimer of warranties" in readme

    def test_attribution_without_copyright_notice(self, fxt_demo_files_service: DemoFilesService) -> None:
        attribution = _MIT_ATTRIBUTION.model_copy(update={"copyright": None})

        files = fxt_demo_files_service.build_demo_files(
            project_id=uuid4(),
            model_format=ModelFormat.OPENVINO,
            license=License(name="MIT", url="https://example.com/license"),
            attribution=attribution,
        )

        by_name = {f.name: f.data for f in files}
        assert "Copyright (c) Foo Research Lab" in by_name["LICENSE"].decode("utf-8")
        assert "The original work does not provide a copyright notice." in by_name["README.md"].decode("utf-8")

    def test_missing_attribution_falls_back_to_generic_notice(self, fxt_demo_files_service: DemoFilesService) -> None:
        files = fxt_demo_files_service.build_demo_files(
            project_id=uuid4(),
            model_format=ModelFormat.OPENVINO,
            license=License(name="MIT", url="https://example.com/license"),
        )

        by_name = {f.name: f.data for f in files}
        license_text = by_name["LICENSE"].decode("utf-8")
        readme = by_name["README.md"].decode("utf-8")
        assert "Copyright (c) the authors of the original model and its pretrained weights" in license_text
        assert "<copyright notice>" not in license_text
        assert "### Attribution" not in readme
        assert "### Modifications" in readme

    @pytest.mark.parametrize("license_name", sorted(_LICENSE_FILES_BY_NAME))
    def test_no_placeholder_left_in_license_files(self, license_name: str) -> None:
        for data in load_license_files(license_name, attribution=_MIT_ATTRIBUTION).values():
            assert b"<copyright notice>" not in data
