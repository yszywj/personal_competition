"""Regression tests for the host-side multi-scenario Docker launcher."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from personal_train import train_r9_multi_scenario as multi


TIMESTAMP = "20260907_150000_123456"


def _checkpoint(scenario: str, *, root: Path | None = None) -> multi.ScenarioCheckpoint:
    base = root or multi.PERSONAL_ROOT / "models" / "source_batch"
    return multi.ScenarioCheckpoint(
        path=base / scenario / "best.pt",
        recorded_score=80.5,
        selection="test",
        scenario=scenario,
        source_batch="source_batch",
        latest_round=100,
        best_round=75,
        interrupted=scenario.startswith("H"),
        sha256="a" * 64,
    )


def _write_resume_source(
    root: Path,
    scenarios,
    *,
    mismatch: str | None = None,
) -> tuple[Path, Path]:
    models_root = root / "models"
    results_root = root / "results"
    batch = models_root / "source_batch"
    for scenario in scenarios:
        model_dir = batch / scenario
        result_dir = results_root / "results_past" / "source_batch" / scenario
        model_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        (model_dir / "best.pt").write_bytes(f"checkpoint-{scenario}".encode())
        (model_dir / "model_metadata.json").write_text(
            json.dumps(
                {
                    "checkpoint_schema": {
                        "name": multi.CURRENT_SCHEMA_NAME,
                        "version": multi.CURRENT_SCHEMA_VERSION,
                    },
                    "algorithm": multi.CURRENT_ALGORITHM,
                    "latest_round": 40 if scenario.startswith("H") else 100,
                    "best_round": 38 if scenario.startswith("H") else 75,
                    "best_official_score": 80.5,
                    "interrupted": scenario.startswith("H"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (result_dir / "run_config.json").write_text(
            json.dumps({"scenario": mismatch if mismatch == scenario else scenario, "rounds": 100})
            + "\n",
            encoding="utf-8",
        )
    return models_root, results_root


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = _checkpoint("E01")

    def test_launcher_accepts_r9_satellite_contract_schema_v3(self):
        self.assertEqual(multi.CURRENT_SCHEMA_VERSION, 3)

    def test_numeric_container_uid_does_not_require_a_passwd_entry(self):
        with mock.patch.object(multi.pwd, "getpwuid", side_effect=KeyError):
            self.assertEqual(multi._username_for_uid(1011), "uid1011")

    def test_host_username_is_preserved(self):
        with mock.patch.object(
            multi.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="trainer")
        ):
            self.assertEqual(multi._username_for_uid(1011), "trainer")

    def test_resume_batch_plan_has_nine_flat_named_jobs_and_all_resume(self):
        checkpoints = {scenario: _checkpoint(scenario) for scenario in multi.ALL_SCENARIOS}
        plans = multi.build_job_plans(
            multi.ALL_SCENARIOS,
            TIMESTAMP,
            checkpoints,
        )

        self.assertEqual([plan.scenario for plan in plans], list(multi.ALL_SCENARIOS))
        self.assertEqual(len(plans), 9)
        self.assertTrue(all(plan.resume == checkpoints[plan.scenario].path for plan in plans))
        self.assertTrue(
            all(
                plan.result_dir
                == multi.RESULTS_ROOT / f"{plan.scenario.lower()}_r9_ppo_{TIMESTAMP}"
                for plan in plans
            )
        )
        self.assertTrue(
            all(
                plan.model_dir
                == multi.MODELS_ROOT / f"{plan.scenario.lower()}_r9_ppo_{TIMESTAMP}"
                for plan in plans
            )
        )
        self.assertTrue(all(plan.result_dir.name == plan.model_dir.name for plan in plans))

    def test_parser_defaults_to_two_gpu_slots_and_rejects_accidental_sharing(self):
        args = multi.parse_args([])
        self.assertEqual(args.rounds, 100)
        self.assertEqual(args.devices, ("0", "1"))
        self.assertEqual(args.max_parallel, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(["--gpu-ids", "0,1", "--max-parallel", "3"])
        with self.assertRaises(ValueError):
            multi.parse_devices("0,00")

    def test_docker_job_has_private_native_results_and_one_visible_gpu(self):
        args = multi.parse_args(
            [
                "--scenarios",
                "E01",
                "--batch-id",
                "r9_multi_test",
                "--gpu-ids",
                "1",
            ]
        )
        args.batch_id = "r9_multi_test"
        plan = multi.build_job_plans(("E01",), TIMESTAMP, self.checkpoint)[0]
        command = multi.docker_command(
            args=args,
            plan=plan,
            device="1",
            uid=123,
            gid=456,
            username="trainer",
        )

        self.assertIn("/app/Results:rw,nosuid,nodev,uid=123,gid=456,mode=0775", command)
        self.assertEqual(command[command.index("--gpus") + 1], "device=1")
        trainer_device_index = command.index("--device")
        self.assertEqual(command[trainer_device_index + 1], "cuda:0")
        self.assertEqual(command[command.index("--scenario") + 1], "E01")
        self.assertIn("--resume", command)
        self.assertEqual(
            command[command.index("--result-dir") + 1],
            f"/app/personal_train/results/e01_r9_ppo_{TIMESTAMP}",
        )
        self.assertEqual(command[command.index("--run-id") + 1], plan.run_name)
        self.assertEqual(
            command[command.index("--cidfile") + 1],
            str(multi.LAUNCHER_RUNS_ROOT / "r9_multi_test" / "launcher_logs" / "E01.cid"),
        )
        self.assertIn("COMPETITION_REPO_ROOT=/opt/competition-platform-env", command)
        self.assertIn("PERSONAL_NATIVE_RESULTS=/app/Results", command)
        self.assertIn(
            f"type=bind,src={multi.REPOSITORY_ROOT.resolve()},dst=/opt/competition-platform-env,readonly",
            command,
        )
        self.assertIn(
            f"type=bind,src={multi.PERSONAL_ROOT.resolve()},dst=/app/personal_train",
            command,
        )
        self.assertNotIn("-v", command)

    def test_cpu_job_omits_docker_gpu_option(self):
        args = multi.parse_args(
            ["--scenarios", "E02", "--batch-id", "cpu_test", "--gpu-ids", "cpu"]
        )
        args.batch_id = "cpu_test"
        plan = multi.build_job_plans(("E02",), TIMESTAMP, None)[0]
        command = multi.docker_command(
            args=args,
            plan=plan,
            device="cpu",
            uid=123,
            gid=456,
            username="trainer",
        )

        self.assertNotIn("--gpus", command)
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertNotIn("--resume", command)

    def test_existing_flat_output_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ), mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                plans = multi.build_job_plans(("E02",), TIMESTAMP, None)
                plans[0].result_dir.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    multi.validate_batch_outputs("r9_multi_test", plans)

    def test_flat_output_symlink_is_rejected_before_any_directory_is_created(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            (root / "redirect").symlink_to(multi.REPOSITORY_ROOT, target_is_directory=True)
            for redirected in ("RESULTS_ROOT", "MODELS_ROOT"):
                with self.subTest(redirected=redirected), mock.patch.object(
                    multi, "RESULTS_ROOT", root / "results"
                ), mock.patch.object(multi, "MODELS_ROOT", root / "models"), mock.patch.object(
                    multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"
                ), mock.patch.object(multi, redirected, root / "redirect"):
                    plans = multi.build_job_plans(("E02",), TIMESTAMP, None)
                    with self.assertRaises(ValueError):
                        multi.validate_batch_outputs("must-not-create", plans)
            self.assertFalse((root / "results").exists())
            self.assertFalse((root / "models").exists())

    def test_resume_batch_accepts_interrupted_source_and_records_provenance(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            models_root, results_root = _write_resume_source(root, ("E01", "H01"))
            with mock.patch.object(multi, "MODELS_ROOT", models_root), mock.patch.object(
                multi, "RESULTS_ROOT", results_root
            ):
                selected = multi.resolve_resume_batch(
                    Path("source_batch"), ("E01", "H01"), root
                )
                selected_by_path = multi.resolve_resume_batch(
                    models_root / "source_batch", ("E01", "H01"), root
                )

            self.assertEqual(set(selected), {"E01", "H01"})
            self.assertEqual(selected_by_path["E01"].path, selected["E01"].path)
            self.assertTrue(selected["H01"].interrupted)
            self.assertEqual(selected["H01"].best_round, 38)
            self.assertRegex(selected["H01"].sha256 or "", r"^[0-9a-f]{64}$")

    def test_resume_batch_rejects_scenario_mismatch(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            models_root, results_root = _write_resume_source(root, ("E01",), mismatch="E01")
            config_path = results_root / "results_past" / "source_batch" / "E01" / "run_config.json"
            config_path.write_text(json.dumps({"scenario": "E02", "rounds": 100}), encoding="utf-8")
            with mock.patch.object(multi, "MODELS_ROOT", models_root), mock.patch.object(
                multi, "RESULTS_ROOT", results_root
            ), self.assertRaises(ValueError):
                multi.resolve_resume_batch(Path("source_batch"), ("E01",), root)

    def test_resume_batch_rejects_schema_mismatch(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            models_root, results_root = _write_resume_source(root, ("E01",))
            metadata_path = models_root / "source_batch" / "E01" / "model_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["checkpoint_schema"]["version"] = multi.CURRENT_SCHEMA_VERSION - 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with mock.patch.object(multi, "MODELS_ROOT", models_root), mock.patch.object(
                multi, "RESULTS_ROOT", results_root
            ), self.assertRaises(ValueError):
                multi.resolve_resume_batch(Path("source_batch"), ("E01",), root)


class BatchReportingTests(unittest.TestCase):
    def test_batch_summary_keeps_initial_and_new_scores_distinct(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            result_dir = root / "results" / "E01"
            result_dir.mkdir(parents=True)
            with (result_dir / "round_scores.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=("round", "score"))
                writer.writeheader()
                writer.writerows(({"round": 1, "score": 70.0}, {"round": 2, "score": 79.0}))
            plan = multi.JobPlan(
                scenario="E01",
                result_dir=result_dir,
                model_dir=root / "models" / "E01",
                resume=multi.PERSONAL_ROOT / "models" / "old" / "best.pt",
                initial_score=80.5,
            )
            states = multi._empty_state((plan,))
            states["E01"].update({"status": "succeeded", "return_code": 0})

            rows = multi.batch_summary((plan,), states)
            self.assertEqual(rows[0]["rounds_completed"], 2)
            self.assertEqual(rows[0]["best_score_in_batch"], 79.0)
            self.assertEqual(rows[0]["final_score"], 79.0)
            self.assertEqual(rows[0]["best_score_including_initial"], 80.5)

            multi.write_batch_reports(root, rows)
            self.assertTrue((root / "batch_scores.txt").is_file())
            self.assertTrue((root / "batch_summary.csv").is_file())
            self.assertTrue((root / "batch_summary.json").is_file())
            self.assertIn("DejaVu Sans", (root / "batch_dashboard.svg").read_text(encoding="utf-8"))


class _ImmediateProcess:
    next_pid = 2000

    def __init__(self, command, **_kwargs):
        type(self).next_pid += 1
        self.command = list(command)
        self.pid = type(self).next_pid
        self.return_code = 0

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        return self.return_code

    def terminate(self):
        self.return_code = -15

    def kill(self):
        self.return_code = -9


class _DelayedProcess(_ImmediateProcess):
    def __init__(self, command, **kwargs):
        super().__init__(command, **kwargs)
        self.return_code = None
        self.poll_count = 0
        self.finished = False

    def poll(self):
        if self.finished:
            return 0
        self.poll_count += 1
        if self.poll_count >= 2:
            self.finished = True
            self.return_code = 0
        return self.return_code


def _write_fake_success(command, rounds=1):
    result_value = command[command.index("--result-dir") + 1]
    model_value = command[command.index("--model-dir") + 1]
    result_relative = Path(result_value).relative_to("/app/personal_train")
    model_relative = Path(model_value).relative_to("/app/personal_train")
    result_dir = multi.PERSONAL_ROOT / result_relative
    model_dir = multi.PERSONAL_ROOT / model_relative
    result_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    scenario = command[command.index("--scenario") + 1]
    with (result_dir / "round_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("round", "score"))
        writer.writeheader()
        for round_index in range(1, rounds + 1):
            writer.writerow({"round": round_index, "score": 75.0})
    for filename in (
        "round_scores.txt",
        "round_scores.svg",
        "training_dashboard.svg",
    ):
        (result_dir / filename).write_text("test\n", encoding="utf-8")
    (result_dir / "run_config.json").write_text(
        json.dumps({"scenario": scenario, "rounds": rounds}) + "\n",
        encoding="utf-8",
    )
    for filename in ("latest.pt", "best.pt"):
        (model_dir / filename).write_bytes(b"test")
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_schema": {
                    "name": multi.CURRENT_SCHEMA_NAME,
                    "version": multi.CURRENT_SCHEMA_VERSION,
                },
                "algorithm": multi.CURRENT_ALGORITHM,
                "latest_round": rounds,
                "best_round": 1,
                "best_official_score": 75.0,
                "interrupted": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (model_dir / "checkpoint_validation.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "algorithm": multi.CURRENT_ALGORITHM,
                "checkpoints": {
                    filename: {
                        "file": filename,
                        "size_bytes": (model_dir / filename).stat().st_size,
                        "sha256": multi._sha256_file(model_dir / filename),
                    }
                    for filename in ("best.pt", "latest.pt")
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


class SchedulerSmokeTests(unittest.TestCase):
    def test_output_metadata_best_fields_must_match_recorded_rounds(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            plan = multi.JobPlan("E02", root / "results", root / "models")
            command = [
                "--scenario",
                "E02",
                "--result-dir",
                multi._container_personal_path(plan.result_dir),
                "--model-dir",
                multi._container_personal_path(plan.model_dir),
            ]
            _write_fake_success(command)
            metadata_path = plan.model_dir / "model_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["best_official_score"] = 99.0
            metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

            errors = multi.validate_job_outputs(plan, 1)

            self.assertTrue(any("does not match round_scores.csv" in error for error in errors))

    def test_tampered_checkpoint_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            plan = multi.JobPlan("E02", root / "results", root / "models")
            command = [
                "--scenario",
                "E02",
                "--result-dir",
                multi._container_personal_path(plan.result_dir),
                "--model-dir",
                multi._container_personal_path(plan.model_dir),
            ]
            _write_fake_success(command)
            (plan.model_dir / "best.pt").write_bytes(b"tampered")

            errors = multi.validate_job_outputs(plan, 1)

            self.assertTrue(any("checkpoint_validation.json" in error for error in errors))

    def test_graceful_stop_targets_owned_cid_with_supported_docker_flags(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            args = multi.parse_args(
                ["--scenarios", "E02", "--batch-id", "stop_test", "--gpu-ids", "0"]
            )
            args.batch_id = "stop_test"
            cidfile = root / "E02.cid"
            cidfile.write_text("a" * 64 + "\n", encoding="utf-8")
            plan = multi.JobPlan("E02", root / "results", root / "models")
            process = _ImmediateProcess(("docker",))
            running = multi.RunningJob(
                plan=plan,
                device="0",
                slot=0,
                container_name="r9-stop-test",
                cidfile=cidfile,
                process=process,
                log_stream=io.StringIO(),
                started_monotonic=0.0,
            )
            stopped = SimpleNamespace(returncode=0)
            absent = SimpleNamespace(returncode=1, stdout="", stderr="Error: No such object")
            with mock.patch.object(multi.subprocess, "run", side_effect=(stopped, absent)) as run:
                multi._stop_containers(args, {"E02": running})

            command = run.call_args_list[0].args[0]
            self.assertEqual(
                command,
                ["docker", "stop", "--signal", "TERM", "--timeout", "60", "a" * 64],
            )

    def test_failed_docker_create_does_not_stop_a_nonexistent_or_foreign_name(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            args = multi.parse_args(
                ["--scenarios", "E02", "--batch-id", "absent_test", "--gpu-ids", "0"]
            )
            args.batch_id = "absent_test"
            running = multi.RunningJob(
                plan=multi.JobPlan("E02", root / "results", root / "models"),
                device="0",
                slot=0,
                container_name="r9-absent-test",
                cidfile=root / "missing.cid",
                process=_ImmediateProcess(("docker",)),
                log_stream=io.StringIO(),
                started_monotonic=0.0,
            )
            absent = SimpleNamespace(returncode=1, stdout="", stderr="Error: No such object")
            no_labels = SimpleNamespace(returncode=0, stdout="", stderr="")

            def inspect_or_list(command, **_kwargs):
                return absent if command[1] == "inspect" else no_labels

            with mock.patch.object(multi.subprocess, "run", side_effect=inspect_or_list) as run, mock.patch.object(
                multi.time, "sleep"
            ):
                self.assertTrue(multi._stop_containers(args, {"E02": running}))
            verbs = [call.args[0][1] for call in run.call_args_list]
            self.assertEqual(verbs, ["inspect", "ps", "inspect", "ps"])

    def test_same_batch_labels_with_a_different_owner_are_never_stopped(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            args = multi.parse_args(
                ["--scenarios", "E02", "--batch-id", "shared_name", "--gpu-ids", "0"]
            )
            args.batch_id = "shared_name"
            running = multi.RunningJob(
                plan=multi.JobPlan("E02", root / "results", root / "models"),
                device="0",
                slot=0,
                container_name="r9-shared-name",
                cidfile=root / "missing.cid",
                process=_ImmediateProcess(("docker",)),
                log_stream=io.StringIO(),
                started_monotonic=0.0,
            )
            foreign = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "personal_train.batch": args.batch_id,
                        "personal_train.scenario": "E02",
                        "personal_train.owner": "another-launcher",
                    }
                ),
                stderr="",
            )

            with mock.patch.object(multi.subprocess, "run", return_value=foreign) as run:
                self.assertTrue(multi._stop_containers(args, {"E02": running}))
            self.assertTrue(all(call.args[0][1] == "inspect" for call in run.call_args_list))

    def test_mocked_batch_creates_common_roots_and_scenario_leaves(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            results_root = root / "results"
            models_root = root / "models"
            launcher_runs_root = root / "launcher_runs"
            args = multi.parse_args(
                [
                    "--scenarios",
                    "E01",
                    "E02",
                    "--rounds",
                    "1",
                    "--batch-id",
                    "scheduler_test",
                    "--gpu-ids",
                    "0,1",
                ]
            )
            args.batch_id = "scheduler_test"
            source = root / "source" / "E01" / "best.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source checkpoint")
            checkpoint = multi.E01Checkpoint(
                path=source,
                recorded_score=80.5,
                selection="test",
                scenario="E01",
                source_batch="source",
                latest_round=100,
                best_round=72,
            )
            with mock.patch.object(multi, "RESULTS_ROOT", results_root), mock.patch.object(
                multi, "MODELS_ROOT", models_root
            ), mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", launcher_runs_root):
                plans = multi.build_job_plans(("E01", "E02"), TIMESTAMP, checkpoint)
                commands = []

                def factory(command, **kwargs):
                    commands.append(list(command))
                    _write_fake_success(command)
                    return _ImmediateProcess(command, **kwargs)

                with mock.patch.object(multi.subprocess, "Popen", side_effect=factory), mock.patch.object(
                    multi.time, "sleep"
                ), contextlib.redirect_stdout(io.StringIO()):
                    return_code = multi.run_batch(args, plans, checkpoint)

            self.assertEqual(return_code, 0)
            self.assertEqual(len(commands), 2)
            self.assertIn("--resume", commands[0])
            self.assertNotIn("--resume", commands[1])
            batch_root = launcher_runs_root / "scheduler_test"
            status = json.loads((batch_root / "batch_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["counts"], {"succeeded": 2})
            self.assertTrue((batch_root / "launcher_logs" / "E01.log").is_file())
            self.assertTrue((batch_root / "batch_dashboard.svg").is_file())
            self.assertTrue((models_root / f"e01_r9_ppo_{TIMESTAMP}").is_dir())
            manifest = json.loads((batch_root / "batch_config.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["resume_checkpoints"]["E01"]["path"], str(source))
            self.assertEqual(manifest["resume_checkpoints"]["E01"]["recorded_best_round"], 72)
            self.assertRegex(manifest["resume_checkpoints"]["E01"]["sha256"], r"^[0-9a-f]{64}$")

    def test_zero_exit_without_required_outputs_is_a_batch_failure(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            args = multi.parse_args(
                ["--scenarios", "E02", "--rounds", "1", "--batch-id", "missing_test", "--gpu-ids", "0"]
            )
            args.batch_id = "missing_test"
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ), mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                plans = multi.build_job_plans(("E02",), TIMESTAMP, None)
                with mock.patch.object(multi.subprocess, "Popen", _ImmediateProcess), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    return_code = multi.run_batch(args, plans, None)

            self.assertEqual(return_code, 1)
            status = json.loads(
                ((root / "launcher_runs" / "missing_test" / "batch_status.json").read_text(encoding="utf-8"))
            )
            self.assertEqual(status["jobs"]["E02"]["status"], "failed")
            self.assertIn("missing required outputs", status["jobs"]["E02"]["error"])

    def test_failed_worker_does_not_prevent_the_next_scenario(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            args = multi.parse_args(
                [
                    "--scenarios",
                    "E02",
                    "E03",
                    "--rounds",
                    "1",
                    "--batch-id",
                    "failure_test",
                    "--gpu-ids",
                    "0",
                ]
            )
            args.batch_id = "failure_test"
            call_index = 0

            def factory(command, **kwargs):
                nonlocal call_index
                call_index += 1
                process = _ImmediateProcess(command, **kwargs)
                if call_index == 1:
                    process.return_code = 17
                else:
                    _write_fake_success(command)
                return process

            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ), mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                plans = multi.build_job_plans(("E02", "E03"), TIMESTAMP, None)
                with mock.patch.object(multi.subprocess, "Popen", side_effect=factory), mock.patch.object(
                    multi, "_stop_containers"
                ), contextlib.redirect_stdout(io.StringIO()):
                    return_code = multi.run_batch(args, plans, None)

            status = json.loads(
                ((root / "launcher_runs" / "failure_test" / "batch_status.json").read_text(encoding="utf-8"))
            )
            self.assertEqual(return_code, 1)
            self.assertEqual(call_index, 2)
            self.assertEqual(status["jobs"]["E02"]["status"], "failed")
            self.assertEqual(status["jobs"]["E02"]["return_code"], 17)
            self.assertEqual(status["jobs"]["E03"]["status"], "succeeded")

    def test_popen_error_is_recorded_and_later_scenario_still_runs(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            args = multi.parse_args(
                [
                    "--scenarios",
                    "E02",
                    "E03",
                    "--rounds",
                    "1",
                    "--batch-id",
                    "popen_test",
                    "--gpu-ids",
                    "0",
                ]
            )
            args.batch_id = "popen_test"
            call_index = 0

            def factory(command, **kwargs):
                nonlocal call_index
                call_index += 1
                if call_index == 1:
                    raise FileNotFoundError("test docker missing")
                _write_fake_success(command)
                return _ImmediateProcess(command, **kwargs)

            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ), mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                plans = multi.build_job_plans(("E02", "E03"), TIMESTAMP, None)
                with mock.patch.object(multi.subprocess, "Popen", side_effect=factory), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    return_code = multi.run_batch(args, plans, None)

            status = json.loads(
                ((root / "launcher_runs" / "popen_test" / "batch_status.json").read_text(encoding="utf-8"))
            )
            self.assertEqual(return_code, 1)
            self.assertEqual(status["jobs"]["E02"]["return_code"], 127)
            self.assertIn("FileNotFoundError", status["jobs"]["E02"]["error"])
            self.assertEqual(status["jobs"]["E03"]["status"], "succeeded")

    def test_scheduler_never_exceeds_two_active_slots(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            args = multi.parse_args(
                [
                    "--scenarios",
                    "E02",
                    "E03",
                    "M01",
                    "--rounds",
                    "1",
                    "--batch-id",
                    "parallel_test",
                    "--gpu-ids",
                    "0,1",
                    "--max-parallel",
                    "2",
                ]
            )
            args.batch_id = "parallel_test"
            processes = []
            maximum_live = 0

            def factory(command, **kwargs):
                nonlocal maximum_live
                live = sum(not process.finished for process in processes)
                maximum_live = max(maximum_live, live + 1)
                _write_fake_success(command)
                process = _DelayedProcess(command, **kwargs)
                processes.append(process)
                return process

            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ), mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                plans = multi.build_job_plans(("E02", "E03", "M01"), TIMESTAMP, None)
                with mock.patch.object(multi.subprocess, "Popen", side_effect=factory), mock.patch.object(
                    multi.time, "sleep"
                ), contextlib.redirect_stdout(io.StringIO()):
                    return_code = multi.run_batch(args, plans, None)

            self.assertEqual(return_code, 0)
            self.assertEqual(maximum_live, 2)

    def test_sigterm_cancels_pending_and_marks_active_interrupted(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            args = multi.parse_args(
                [
                    "--scenarios",
                    "E02",
                    "E03",
                    "--rounds",
                    "1",
                    "--batch-id",
                    "signal_test",
                    "--gpu-ids",
                    "0",
                ]
            )
            args.batch_id = "signal_test"
            processes = []

            class WaitingProcess(_ImmediateProcess):
                def __init__(self, command, **kwargs):
                    super().__init__(command, **kwargs)
                    self.return_code = None

                def poll(self):
                    return self.return_code

            def factory(command, **kwargs):
                process = WaitingProcess(command, **kwargs)
                processes.append(process)
                return process

            signal_sent = False

            def trigger_signal(_seconds):
                nonlocal signal_sent
                if not signal_sent:
                    signal_sent = True
                    os.kill(os.getpid(), signal.SIGTERM)

            def fake_stop(_args, active):
                for running in active.values():
                    running.process.return_code = 130
                return True

            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ), mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                plans = multi.build_job_plans(("E02", "E03"), TIMESTAMP, None)
                with mock.patch.object(multi.subprocess, "Popen", side_effect=factory), mock.patch.object(
                    multi, "_stop_containers", side_effect=fake_stop
                ), mock.patch.object(multi.time, "sleep", side_effect=trigger_signal), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    return_code = multi.run_batch(args, plans, None)

            self.assertEqual(return_code, 130)
            status = json.loads(
                ((root / "launcher_runs" / "signal_test" / "batch_status.json").read_text(encoding="utf-8"))
            )
            self.assertTrue(status["interrupted"])
            self.assertEqual(status["jobs"]["E02"]["status"], "interrupted")
            self.assertEqual(status["jobs"]["E03"]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
