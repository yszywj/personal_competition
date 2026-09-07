"""Infrastructure-only tests; no simulator, policy update or GPU is started."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from personal_train import bootstrap
from personal_train.device import resolve_learning_device


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=bootstrap.PERSONAL_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_source(self, root: Path, *, flat: bool = False) -> Path:
        core = root if flat else root / "core"
        (core / "envengine").mkdir(parents=True)
        (root / "policies").mkdir()
        (root / "scenarios" / "cases").mkdir(parents=True)
        return core

    def test_sibling_checkout(self):
        repository = self.root / "competition-platform-env"
        core = self.make_source(repository)
        layout = bootstrap.discover_project_layout(self.root / "personal_train")
        self.assertEqual(layout.repository_root, repository)
        self.assertEqual(layout.core_root, core)
        self.assertEqual(layout.scenarios_root, repository / "scenarios")

    def test_legacy_nested_checkout(self):
        core = self.make_source(self.root)
        layout = bootstrap.discover_project_layout(self.root / "personal_train")
        self.assertEqual(layout.core_root, core)
        self.assertEqual(layout.repository_root, self.root)

    def test_flat_container_checkout(self):
        self.make_source(self.root, flat=True)
        layout = bootstrap.discover_project_layout(self.root / "personal_train")
        self.assertEqual(layout.core_root, self.root)

    def test_explicit_source_takes_precedence(self):
        self.make_source(self.root / "competition-platform-env")
        other = self.root / "selected-upstream"
        self.make_source(other)
        layout = bootstrap.discover_project_layout(self.root / "personal_train", other)
        self.assertEqual(layout.repository_root, other)

    def test_invalid_explicit_source_does_not_fall_back(self):
        self.make_source(self.root / "competition-platform-env")
        with self.assertRaisesRegex(FileNotFoundError, "COMPETITION_REPO_ROOT"):
            bootstrap.discover_project_layout(
                self.root / "personal_train", self.root / "missing"
            )

    def test_missing_case_package_is_rejected(self):
        (self.root / "envengine").mkdir()
        (self.root / "policies").mkdir()
        with self.assertRaises(FileNotFoundError):
            bootstrap.discover_project_layout(self.root / "personal_train")

    def test_source_precedes_baked_image_and_install_is_idempotent(self):
        with mock.patch.object(sys, "path", ["/app", "existing"]), mock.patch.object(
            sys, "dont_write_bytecode", False
        ):
            bootstrap.install_project_paths()
            first = list(sys.path)
            bootstrap.install_project_paths()
            self.assertEqual(first, sys.path)
            self.assertEqual(sys.path[0], str(bootstrap.CORE_ROOT))
            self.assertLess(sys.path.index(str(bootstrap.REPOSITORY_ROOT)), sys.path.index("/app"))
            self.assertTrue(sys.dont_write_bytecode)

    def test_private_runtime_links_resources_but_keeps_output_separate(self):
        core = self.make_source(self.root / "upstream")
        original_results = core / "Results"
        original_results.mkdir()
        (original_results / "keep.txt").write_text("original", encoding="utf-8")
        runtime = bootstrap.prepare_runtime_directory(self.root / "runtime", core_root=core)
        self.assertTrue((runtime / "envengine").is_symlink())
        self.assertEqual((runtime / "envengine").resolve(), core / "envengine")
        self.assertFalse((runtime / "Results").is_symlink())
        (runtime / "Results" / "new.txt").write_text("output", encoding="utf-8")
        self.assertEqual(sorted(path.name for path in original_results.iterdir()), ["keep.txt"])

    def test_native_results_mount_and_existing_runtime_protection(self):
        core = self.make_source(self.root / "upstream")
        native_results = self.root / "tmpfs"
        native_results.mkdir()
        runtime = bootstrap.prepare_runtime_directory(
            self.root / "runtime", core_root=core, native_results=native_results
        )
        self.assertEqual((runtime / "Results").resolve(), native_results)
        with self.assertRaises(FileExistsError):
            bootstrap.prepare_runtime_directory(runtime, core_root=core)

    def test_runtime_cannot_be_inside_upstream_core(self):
        core = self.make_source(self.root / "upstream")
        with self.assertRaises(ValueError):
            bootstrap.prepare_runtime_directory(core / "runtime", core_root=core)

    def test_missing_native_mount_fails_before_creating_runtime(self):
        core = self.make_source(self.root / "upstream")
        runtime = self.root / "runtime"
        with self.assertRaises(FileNotFoundError):
            bootstrap.prepare_runtime_directory(
                runtime, core_root=core, native_results=self.root / "missing"
            )
        self.assertFalse(runtime.exists())

    def test_output_cannot_be_the_personal_root_or_outside_it(self):
        for path in (bootstrap.PERSONAL_ROOT, bootstrap.PERSONAL_ROOT.parent / "not-output"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                bootstrap.validate_personal_output_path(path)

    def test_result_or_model_symlink_cannot_escape_into_upstream(self):
        for name in ("results", "models"):
            alias = self.root / name
            alias.symlink_to(bootstrap.REPOSITORY_ROOT, target_is_directory=True)
            with self.subTest(name=name), self.assertRaises(ValueError):
                bootstrap.validate_personal_output_path(alias / "new-output")

    def test_valid_output_is_only_resolved_not_created(self):
        output = self.root / "result" / "new"
        self.assertEqual(bootstrap.validate_personal_output_path(output), output)
        self.assertFalse(output.exists())


class DeviceTests(unittest.TestCase):
    def test_auto_and_cpu(self):
        self.assertEqual(resolve_learning_device("auto", cuda_available=False, cuda_device_count=0), "cpu")
        self.assertEqual(resolve_learning_device("auto", cuda_available=True, cuda_device_count=2), "cuda")
        self.assertEqual(resolve_learning_device("cpu", cuda_available=True, cuda_device_count=2), "cpu")

    def test_explicit_gpu_is_not_silently_downgraded(self):
        with self.assertRaises(RuntimeError):
            resolve_learning_device("cuda", cuda_available=False, cuda_device_count=0)
        with self.assertRaises(RuntimeError):
            resolve_learning_device("cuda:2", cuda_available=True, cuda_device_count=2)
        self.assertEqual(resolve_learning_device("cuda:1", cuda_available=True, cuda_device_count=2), "cuda:1")

    def test_invalid_device(self):
        with self.assertRaises(ValueError):
            resolve_learning_device("cuda:abc", cuda_available=True, cuda_device_count=2)


if __name__ == "__main__":
    unittest.main()
