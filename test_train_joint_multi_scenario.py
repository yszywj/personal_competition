"""CPU-only regression tests for the joint-PPO multi-scenario launcher."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from personal_train import train_joint_multi_scenario as multi


TIMESTAMP = "20260910_120000_123456"


def _write_success_outputs(command: list[str], rounds: int = 1) -> None:
    result_dir = Path(command[command.index("--result-dir") + 1])
    model_dir = Path(command[command.index("--model-dir") + 1])
    selector = command[command.index("--scenario") + 1]
    scenario = Path(selector).name.upper()
    suite = selector.split("/", 1)[0] if "/" in selector else "legacy"
    scenario_path = multi.scenario_file_path(multi.REPOSITORY_ROOT, suite, scenario)
    result_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "scenario": scenario,
                "scenario_path": str(scenario_path),
                "rounds": rounds,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (result_dir / "status.json").write_text(
        json.dumps({"status": "complete", "completed_rounds": rounds}) + "\n",
        encoding="utf-8",
    )
    rows = ["round,score"] + [f"{index},75.0" for index in range(1, rounds + 1)]
    (result_dir / "rounds.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (model_dir / "latest.pt").write_bytes(b"latest")
    (model_dir / "best.pt").write_bytes(b"best")
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": multi.JOINT_ALGORITHM,
                "schema_version": multi.JOINT_CHECKPOINT_SCHEMA_VERSION,
                "latest_round": rounds,
                "status": "complete",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoints = {}
    for name, role, resume_safe in (
        ("best", "best_behavior", False),
        ("latest", "latest", True),
    ):
        path = model_dir / f"{name}.pt"
        checkpoints[name] = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": multi._sha256_file(path),
            "checkpoint_role": role,
            "resume_safe": resume_safe,
            "update_count": 1,
            "episode_count": rounds,
        }
    (model_dir / "checkpoint_validation.json").write_text(
        json.dumps(
            {
                "algorithm": multi.JOINT_ALGORITHM,
                "schema_version": multi.JOINT_CHECKPOINT_SCHEMA_VERSION,
                "checkpoints": checkpoints,
            }
        )
        + "\n",
        encoding="utf-8",
    )


class _ImmediateSuccess:
    next_pid = 7000
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def __init__(self, command, **kwargs):
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.return_code = 0
        copied = list(command)
        type(self).commands.append(copied)
        type(self).environments.append(dict(kwargs["env"]))
        _write_success_outputs(copied)

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        return self.return_code


class ParserAndPlacementTests(unittest.TestCase):
    def test_devices_are_explicit_and_gpu_zero_is_protected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args([])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(["--gpu-ids", "0"])

        args = multi.parse_args(["--gpu-ids", "0", "--allow-gpu-zero"])
        self.assertEqual(args.devices, ("0",))

    def test_nine_jobs_on_seven_free_gpus_require_explicit_sharing(self):
        repeated = "1,2,3,4,5,6,7,1,2"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(["--gpu-ids", repeated])

        args = multi.parse_args(
            ["--gpu-ids", repeated, "--allow-gpu-sharing"]
        )
        self.assertEqual(len(args.devices), 9)
        self.assertEqual(args.max_parallel, 9)

    def test_nine_jobs_on_all_eight_gpus_need_zero_and_sharing_acknowledgements(self):
        repeated = "0,1,2,3,4,5,6,7,1"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(
                ["--gpu-ids", repeated, "--allow-gpu-sharing"]
            )
        args = multi.parse_args(
            [
                "--gpu-ids",
                repeated,
                "--allow-gpu-zero",
                "--allow-gpu-sharing",
            ]
        )
        self.assertEqual(len(args.devices), 9)
        self.assertEqual(args.max_parallel, 9)

    def test_numa_and_cpu_sets_follow_worker_slots(self):
        args = multi.parse_args(
            [
                "--gpu-ids",
                "1,2",
                "--numa-nodes",
                "0,1",
                "--cpu-sets",
                "0-3",
                "32-35",
                "--threads-per-worker",
                "2",
            ]
        )
        slots = multi.build_worker_slots(args)
        self.assertEqual(slots[0], multi.WorkerSlot(0, "1", 0, "0-3"))
        self.assertEqual(slots[1], multi.WorkerSlot(1, "2", 1, "32-35"))

        plan = multi.build_job_plans(("E01",), TIMESTAMP)[0]
        command = multi.trainer_command(args, plan, slots[1])
        self.assertEqual(
            command[:4],
            ["numactl", "--physcpubind=32-35", "--membind=1", args.python],
        )
        environment = multi.worker_environment(
            args, plan, slots[1], multi.PERSONAL_ROOT / "launcher_runs" / "test", base={}
        )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(environment["OMP_NUM_THREADS"], "2")
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")

    def test_preflight_protects_gpu_zero_selected_by_uuid(self):
        args = multi.parse_args(["--gpu-ids", "GPU-AAAA"])
        slots = multi.build_worker_slots(args)
        completed = mock.Mock(stdout="0, GPU-AAAA\n1, GPU-BBBB\n")
        real_which = multi.shutil.which

        def fake_which(value):
            return "/usr/bin/nvidia-smi" if value == "nvidia-smi" else real_which(value)

        with mock.patch.object(multi.shutil, "which", side_effect=fake_which), mock.patch.object(
            multi.subprocess, "run", return_value=completed
        ), self.assertRaisesRegex(ValueError, "GPU 0 is protected"):
            multi.preflight(args, slots)

    def test_overlapping_cpu_sets_and_protected_trainer_args_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(
                ["--gpu-ids", "1,2", "--cpu-sets", "0-3", "3-7"]
            )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(
                ["--gpu-ids", "1", "--trainer-args", "--device", "cpu"]
            )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(
                ["--gpu-ids", "1", "--trainer-args", "--devi", "cpu"]
            )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(
                ["--gpu-ids", "1", "--trainer-args", "--result-d", "/tmp/escape"]
            )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(
                ["--gpu-ids", "1", "--trainer-args", "--init-f", "/tmp/escape.pt"]
            )

    def test_non_finite_timeouts_are_rejected(self):
        for option, value in (
            ("--stop-timeout", "nan"),
            ("--stop-timeout", "inf"),
            ("--poll-interval", "nan"),
            ("--poll-interval", "inf"),
        ):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                multi.parse_args(["--gpu-ids", "1", option, value])

    def test_suite_defaults_and_case_membership(self):
        legacy = multi.parse_args(["--gpu-ids", "cpu"])
        final24 = multi.parse_args(["--gpu-ids", "cpu", "--suite", "final24"])
        final20 = multi.parse_args(["--gpu-ids", "cpu", "--suite", "final20"])

        self.assertEqual(legacy.suite, "legacy")
        self.assertEqual(legacy.scenarios, multi.ALL_SCENARIOS)
        self.assertEqual(len(final24.scenarios), 24)
        self.assertEqual(len(final20.scenarios), 20)
        self.assertIn("E08", final24.scenarios)
        self.assertNotIn("E07", final20.scenarios)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            multi.parse_args(
                ["--gpu-ids", "cpu", "--suite", "final20", "--scenarios", "E07"]
            )

    def test_final20_plans_use_unambiguous_paths_keys_and_outputs(self):
        plans = multi.build_job_plans(
            ("E01", "M01", "H01"),
            TIMESTAMP,
            suite="final20",
        )
        self.assertEqual(
            [plan.selector for plan in plans],
            ["final20/easy/E01", "final20/medium/M01", "final20/hard/H01"],
        )
        self.assertEqual(
            [plan.job_key for plan in plans],
            ["final20:E01", "final20:M01", "final20:H01"],
        )
        self.assertTrue(
            all(plan.result_dir.name.startswith("final20_") for plan in plans)
        )

        args = multi.parse_args(
            ["--gpu-ids", "cpu", "--suite", "final20", "--scenarios", "E01"]
        )
        command = multi.trainer_command(args, plans[0], multi.build_worker_slots(args)[0])
        self.assertEqual(
            command[command.index("--scenario") + 1],
            "final20/easy/E01",
        )

    def test_all_scenarios_get_independent_flat_outputs(self):
        plans = multi.build_job_plans(multi.ALL_SCENARIOS, TIMESTAMP)
        self.assertEqual(len(plans), 9)
        self.assertEqual(len({plan.result_dir for plan in plans}), 9)
        self.assertEqual(len({plan.model_dir for plan in plans}), 9)
        self.assertTrue(
            all(
                plan.result_dir.name
                == f"{plan.scenario.lower()}_joint_ppo_{TIMESTAMP}"
                for plan in plans
            )
        )


class DryRunAndResumeTests(unittest.TestCase):
    def test_dry_run_does_not_write_or_start_workers(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            launcher_root = root / "launcher_runs"
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ), mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", launcher_root), mock.patch.object(
                multi, "make_timestamp", return_value=TIMESTAMP
            ), mock.patch.object(multi.subprocess, "Popen") as popen, contextlib.redirect_stdout(
                io.StringIO()
            ):
                return_code = multi.main(
                    ["--gpu-ids", "cpu", "--batch-id", "dry_only", "--dry-run"]
                )

            self.assertEqual(return_code, 0)
            popen.assert_not_called()
            self.assertFalse((launcher_root / "dry_only").exists())
            self.assertFalse((root / "results").exists())
            self.assertFalse((root / "models").exists())

    def test_resume_and_warm_start_modes_are_mutually_exclusive(self):
        for values in (
            ["--resume-from", "E01=a.pt", "--init-from", "E01=b.pt"],
            ["--resume-batch", "old_batch", "--init-from", "E01=b.pt"],
        ):
            with self.subTest(values=values), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                multi.parse_args(["--gpu-ids", "cpu", *values])

    def test_per_case_init_from_uses_weights_only_command_and_records_provenance(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            checkpoint = root / "old" / "checkpoints" / "round_0040.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"pre-motion-policy")
            selected = multi.resolve_explicit_initializations(
                [f"E01={checkpoint}"], ("E01",), root
            )
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ):
                plan = multi.build_job_plans(
                    ("E01",), TIMESTAMP, init_froms=selected
                )[0]
            args = multi.parse_args(["--gpu-ids", "cpu", "--scenarios", "E01"])
            slot = multi.build_worker_slots(args)[0]
            command = multi.trainer_command(args, plan, slot)
            source = selected["E01"]

            self.assertEqual(source.mode, "init_from")
            self.assertEqual(source.sha256, multi._sha256_file(checkpoint))
            self.assertEqual(plan.initialization, "weights_only_warm_start")
            self.assertIs(plan.checkpoint_source, source)
            self.assertIsNone(plan.resume)
            self.assertNotIn("--resume", command)
            self.assertEqual(
                command[command.index("--init-from") + 1], str(checkpoint.resolve())
            )
            document = multi.batch_plan_document(
                args,
                (plan,),
                (slot,),
                root / "launcher",
                created_at="2026-09-13T12:00:00+08:00",
            )
            recorded = document["jobs"][0]["checkpoint_source"]
            self.assertEqual(recorded["mode"], "init_from")
            self.assertEqual(recorded["path"], str(checkpoint.resolve()))
            self.assertEqual(recorded["sha256"], source.sha256)
            self.assertIsNone(document["jobs"][0]["resume"])
            self.assertEqual(document["jobs"][0]["init_from"], recorded)
            state = multi._empty_states((plan,))["E01"]
            self.assertEqual(state["checkpoint_source"], recorded)
            self.assertEqual(state["init_from"], str(checkpoint.resolve()))

    def test_init_from_dry_run_is_read_only_and_records_source(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            checkpoint = root / "source" / "round_0040.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"pre-motion")
            stdout = io.StringIO()
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ), mock.patch.object(
                multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"
            ), mock.patch.object(
                multi, "make_timestamp", return_value=TIMESTAMP
            ), mock.patch.object(multi.subprocess, "Popen") as popen, contextlib.redirect_stdout(
                stdout
            ):
                return_code = multi.main(
                    [
                        "--gpu-ids",
                        "cpu",
                        "--scenarios",
                        "E01",
                        "--batch-id",
                        "init_dry",
                        "--init-from",
                        f"E01={checkpoint}",
                        "--dry-run",
                    ]
                )

            self.assertEqual(return_code, 0)
            popen.assert_not_called()
            self.assertFalse((root / "launcher_runs" / "init_dry").exists())
            self.assertFalse((root / "results").exists())
            self.assertFalse((root / "models").exists())
            source = json.loads(stdout.getvalue())["jobs"][0]["checkpoint_source"]
            self.assertEqual(source["mode"], "init_from")
            self.assertEqual(source["path"], str(checkpoint.resolve()))
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")

    def test_warm_start_hash_change_is_rejected_before_popen(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            checkpoint = root / "source" / "round_0040.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"original")
            selected = multi.resolve_explicit_initializations(
                [f"E01={checkpoint}"], ("E01",), root
            )
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ):
                plans = multi.build_job_plans(
                    ("E01",), TIMESTAMP, init_froms=selected
                )
            checkpoint.write_bytes(b"changed-after-planning")
            args = multi.parse_args(
                [
                    "--rounds",
                    "1",
                    "--gpu-ids",
                    "cpu",
                    "--scenarios",
                    "E01",
                    "--batch-id",
                    "changed_source",
                    "--poll-interval",
                    "0.001",
                ]
            )
            args.batch_id = "changed_source"
            launcher = root / "launcher_runs" / "changed_source"
            with mock.patch.object(multi.subprocess, "Popen") as popen, mock.patch.object(
                multi.time, "sleep", return_value=None
            ), contextlib.redirect_stdout(io.StringIO()):
                return_code = multi.run_batch(
                    args, plans, multi.build_worker_slots(args), launcher
                )

            self.assertEqual(return_code, 1)
            popen.assert_not_called()
            status = json.loads(
                (launcher / "batch_status.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                "Warm-start checkpoint changed", status["jobs"]["E01"]["error"]
            )

    def test_resume_batch_selects_each_scenarios_latest_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            launcher = root / "launcher_runs" / "old_batch"
            launcher.mkdir(parents=True)
            jobs = []
            for scenario in ("E01", "M02"):
                model_dir = root / "models" / scenario
                model_dir.mkdir(parents=True)
                (model_dir / "latest.pt").write_bytes(scenario.encode())
                jobs.append({"scenario": scenario, "model_dir": str(model_dir)})
            (launcher / "batch_plan.json").write_text(
                json.dumps(
                    {
                        "schema": multi.LAUNCHER_SCHEMA,
                        # Schema v1 had no suite or selector; it is legacy by definition.
                        "schema_version": 1,
                        "batch_id": "old_batch",
                        "jobs": jobs,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                selected = multi.resolve_resume_batch(
                    Path("old_batch"), ("E01", "M02"), root
                )

            self.assertEqual(set(selected), {"E01", "M02"})
            self.assertEqual(selected["E01"].path, root / "models" / "E01" / "latest.pt")
            self.assertEqual(selected["M02"].source_batch, "old_batch")
            self.assertRegex(selected["E01"].sha256, r"^[0-9a-f]{64}$")

    def test_resume_batch_rejects_same_case_id_from_another_suite(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            launcher = root / "launcher_runs" / "final24_batch"
            model_dir = root / "models" / "final24_e01"
            launcher.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            (model_dir / "latest.pt").write_bytes(b"final24")
            (launcher / "batch_plan.json").write_text(
                json.dumps(
                    {
                        "schema": multi.LAUNCHER_SCHEMA,
                        "schema_version": multi.LAUNCHER_SCHEMA_VERSION,
                        "batch_id": "final24_batch",
                        "suite": "final24",
                        "jobs": [
                            {
                                "scenario": "E01",
                                "suite": "final24",
                                "scenario_selector": "final24/easy/E01",
                                "model_dir": str(model_dir),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                with self.assertRaisesRegex(ValueError, "Prior batch suite is final24"):
                    multi.resolve_resume_batch(
                        Path("final24_batch"),
                        ("E01",),
                        root,
                        suite="final20",
                    )

    def test_resume_batch_rejects_mismatched_scenario_selector(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            launcher = root / "launcher_runs" / "bad_selector"
            model_dir = root / "models" / "final20_e01"
            launcher.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            (model_dir / "latest.pt").write_bytes(b"final20")
            (launcher / "batch_plan.json").write_text(
                json.dumps(
                    {
                        "schema": multi.LAUNCHER_SCHEMA,
                        "schema_version": multi.LAUNCHER_SCHEMA_VERSION,
                        "batch_id": "bad_selector",
                        "suite": "final20",
                        "jobs": [
                            {
                                "scenario": "E01",
                                "suite": "final20",
                                "scenario_selector": "final20/easy/E02",
                                "model_dir": str(model_dir),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                with self.assertRaisesRegex(ValueError, "no plan"):
                    multi.resolve_resume_batch(
                        Path("bad_selector"),
                        ("E01",),
                        root,
                        suite="final20",
                    )

    def test_resume_batch_matches_the_recorded_suite_selector_and_path(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            launcher = root / "launcher_runs" / "final20_batch"
            model_dir = root / "models" / "final20_e01"
            launcher.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            (model_dir / "latest.pt").write_bytes(b"final20")
            scenario_path = multi.scenario_file_path(
                multi.REPOSITORY_ROOT, "final20", "E01"
            )
            plan = {
                "schema": multi.LAUNCHER_SCHEMA,
                "schema_version": multi.LAUNCHER_SCHEMA_VERSION,
                "batch_id": "final20_batch",
                "suite": "final20",
                "jobs": [
                    {
                        "scenario": "E01",
                        "suite": "final20",
                        "scenario_selector": "final20/easy/E01",
                        "scenario_path": str(scenario_path),
                        "model_dir": str(model_dir),
                    }
                ],
            }
            plan_path = launcher / "batch_plan.json"
            plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with mock.patch.object(multi, "LAUNCHER_RUNS_ROOT", root / "launcher_runs"):
                selected = multi.resolve_resume_batch(
                    Path("final20_batch"),
                    ("E01",),
                    root,
                    suite="final20",
                )
                self.assertEqual(selected["E01"].suite, "final20")
                self.assertEqual(
                    selected["E01"].scenario_selector,
                    "final20/easy/E01",
                )

                plan["jobs"][0]["scenario_path"] = str(
                    multi.scenario_file_path(multi.REPOSITORY_ROOT, "legacy", "E01")
                )
                plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "no plan"):
                    multi.resolve_resume_batch(
                        Path("final20_batch"),
                        ("E01",),
                        root,
                        suite="final20",
                    )


class SchedulerTests(unittest.TestCase):
    def test_final20_scheduler_persists_suite_qualified_identity(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            results = root / "results"
            models = root / "models"
            launcher = root / "launcher_runs" / "final20_test"
            with mock.patch.object(multi, "RESULTS_ROOT", results), mock.patch.object(
                multi, "MODELS_ROOT", models
            ):
                plans = multi.build_job_plans(
                    ("E01",),
                    TIMESTAMP,
                    suite="final20",
                )
            args = multi.parse_args(
                [
                    "--rounds",
                    "1",
                    "--gpu-ids",
                    "cpu",
                    "--suite",
                    "final20",
                    "--scenarios",
                    "E01",
                    "--batch-id",
                    "final20_test",
                    "--poll-interval",
                    "0.001",
                ]
            )
            args.batch_id = "final20_test"
            _ImmediateSuccess.commands = []
            _ImmediateSuccess.environments = []
            with mock.patch.object(multi.subprocess, "Popen", _ImmediateSuccess), mock.patch.object(
                multi.time, "sleep", return_value=None
            ), contextlib.redirect_stdout(io.StringIO()):
                return_code = multi.run_batch(
                    args,
                    plans,
                    multi.build_worker_slots(args),
                    launcher,
                )

            self.assertEqual(return_code, 0)
            command = _ImmediateSuccess.commands[0]
            self.assertEqual(
                command[command.index("--scenario") + 1],
                "final20/easy/E01",
            )
            status = json.loads((launcher / "batch_status.json").read_text(encoding="utf-8"))
            self.assertEqual(set(status["jobs"]), {"final20:E01"})
            self.assertTrue((launcher / "logs" / "final20_e01.log").is_file())

    def test_output_validation_rejects_same_id_from_wrong_suite_path(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ):
                plan = multi.build_job_plans(
                    ("E01",),
                    TIMESTAMP,
                    suite="final20",
                )[0]
            command = [
                "python",
                "trainer.py",
                "--scenario",
                plan.selector,
                "--result-dir",
                str(plan.result_dir),
                "--model-dir",
                str(plan.model_dir),
            ]
            _write_success_outputs(command)
            run_config_path = plan.result_dir / "run_config.json"
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            run_config["scenario_path"] = str(
                multi.scenario_file_path(multi.REPOSITORY_ROOT, "legacy", "E01")
            )
            run_config_path.write_text(json.dumps(run_config) + "\n", encoding="utf-8")

            errors = multi.validate_job_outputs(plan, 1)

            self.assertIn(
                "run_config.json scenario path/ID or round count does not match the job",
                errors,
            )

    def test_checkpoint_corruption_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ):
                plan = multi.build_job_plans(("E01",), TIMESTAMP)[0]
            command = [
                "python",
                "trainer.py",
                "--scenario",
                "E01",
                "--result-dir",
                str(plan.result_dir),
                "--model-dir",
                str(plan.model_dir),
            ]
            _write_success_outputs(command)
            (plan.model_dir / "latest.pt").write_bytes(b"tampered")

            errors = multi.validate_job_outputs(plan, 1)

            self.assertTrue(
                any("hash mismatch: latest" in error for error in errors),
                errors,
            )

    def test_non_finite_score_is_rejected_without_breaking_summary(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ):
                plan = multi.build_job_plans(("E01",), TIMESTAMP)[0]
            command = [
                "python",
                "trainer.py",
                "--scenario",
                "E01",
                "--result-dir",
                str(plan.result_dir),
                "--model-dir",
                str(plan.model_dir),
            ]
            _write_success_outputs(command)
            (plan.result_dir / "rounds.csv").write_text(
                "round,score\n1,nan\n", encoding="utf-8"
            )

            errors = multi.validate_job_outputs(plan, 1)
            rows = multi._summary_rows(
                (plan,),
                {
                    "E01": {
                        "status": "failed",
                        "host_device": "cpu",
                        "elapsed_seconds": 1.0,
                    }
                },
            )
            multi._write_summary(root, rows)

            self.assertIn("rounds.csv contains an invalid official score", errors)
            self.assertIsNone(rows[0]["best_score"])
            self.assertIsNone(rows[0]["final_score"])
            json.loads((root / "batch_summary.json").read_text(encoding="utf-8"))

    def test_malformed_worker_documents_are_isolated_as_validation_errors(self):
        for relative_path in (
            Path("result/run_config.json"),
            Path("result/status.json"),
            Path("model/model_metadata.json"),
        ):
            with self.subTest(path=str(relative_path)), tempfile.TemporaryDirectory(
                dir=multi.PERSONAL_ROOT
            ) as directory:
                root = Path(directory)
                with mock.patch.object(multi, "RESULTS_ROOT", root / "result-parent"), mock.patch.object(
                    multi, "MODELS_ROOT", root / "model-parent"
                ):
                    plan = multi.build_job_plans(("E01",), TIMESTAMP)[0]
                command = [
                    "python",
                    "trainer.py",
                    "--scenario",
                    "E01",
                    "--result-dir",
                    str(plan.result_dir),
                    "--model-dir",
                    str(plan.model_dir),
                ]
                _write_success_outputs(command)
                target_root = plan.result_dir if relative_path.parts[0] == "result" else plan.model_dir
                (target_root / relative_path.name).write_text("[]\n", encoding="utf-8")

                errors = multi.validate_job_outputs(plan, 1)

                self.assertTrue(errors)

    def test_unreadable_rounds_csv_is_treated_as_invalid_output(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            with mock.patch.object(multi, "RESULTS_ROOT", root / "results"), mock.patch.object(
                multi, "MODELS_ROOT", root / "models"
            ):
                plan = multi.build_job_plans(("E01",), TIMESTAMP)[0]
            command = [
                "python",
                "trainer.py",
                "--scenario",
                "E01",
                "--result-dir",
                str(plan.result_dir),
                "--model-dir",
                str(plan.model_dir),
            ]
            _write_success_outputs(command)
            (plan.result_dir / "rounds.csv").write_bytes(b"\xff")

            errors = multi.validate_job_outputs(plan, 1)

            self.assertTrue(any("rounds.csv" in error for error in errors), errors)

    def test_nine_cpu_slots_launch_nine_independent_jobs_and_persist_status(self):
        with tempfile.TemporaryDirectory(dir=multi.PERSONAL_ROOT) as directory:
            root = Path(directory)
            results = root / "results"
            models = root / "models"
            launcher = root / "launcher_runs" / "nine_test"
            with mock.patch.object(multi, "RESULTS_ROOT", results), mock.patch.object(
                multi, "MODELS_ROOT", models
            ):
                plans = multi.build_job_plans(multi.ALL_SCENARIOS, TIMESTAMP)
            args = multi.parse_args(
                [
                    "--rounds",
                    "1",
                    "--gpu-ids",
                    ",".join(["cpu"] * 9),
                    "--batch-id",
                    "nine_test",
                    "--poll-interval",
                    "0.001",
                ]
            )
            args.batch_id = "nine_test"
            slots = multi.build_worker_slots(args)
            _ImmediateSuccess.commands = []
            _ImmediateSuccess.environments = []
            with mock.patch.object(multi.subprocess, "Popen", _ImmediateSuccess), mock.patch.object(
                multi.time, "sleep", return_value=None
            ), contextlib.redirect_stdout(io.StringIO()):
                return_code = multi.run_batch(args, plans, slots, launcher)

            self.assertEqual(return_code, 0)
            self.assertEqual(len(_ImmediateSuccess.commands), 9)
            self.assertEqual(
                {command[command.index("--scenario") + 1] for command in _ImmediateSuccess.commands},
                set(multi.ALL_SCENARIOS),
            )
            self.assertTrue(
                all(environment["CUDA_VISIBLE_DEVICES"] == "" for environment in _ImmediateSuccess.environments)
            )
            status = json.loads((launcher / "batch_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(status["counts"], {"succeeded": 9})
            self.assertTrue((launcher / "batch_summary.csv").is_file())
            self.assertTrue((launcher / "batch_plan.json").is_file())


if __name__ == "__main__":
    unittest.main()
