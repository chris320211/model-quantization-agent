"""Typed policies for launching generated quantization code.

Host execution remains the default for backwards compatibility.  A trusted
caller may instead provide a fully-specified container command template.  The
template is rendered into an argv list (never interpolated directly into a
shell command), and the executor quotes each argument before using its small
exit-code wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Mapping


_CONTAINER_RUNTIMES = {"docker", "podman", "nerdctl", "apptainer", "singularity"}
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:token|secret|password|passwd|api[_-]?key|access[_-]?key)\s*="
)


class ExecutionMode(str, Enum):
    HOST = "host"
    CONTAINER = "container"


@dataclass(frozen=True)
class ContainerCommandPlan:
    """Trusted container-runtime argv template.

    Tokens may contain ``{job_dir}``, ``{script_path}``, ``{output_dir}``,
    ``{repo_root}``, ``{job_id}``, ``{method_id}``, and ``{model_id}``.
    This keeps runtime/image/mount/resource policy outside generated code while
    allowing a supported Docker/OCI or HPC container runtime.

    Example::

        ContainerCommandPlan((
            "docker", "run", "--rm", "--gpus", "all",
            "-v", "{job_dir}:/quant-job", "my-image@sha256:...",
            "python", "/quant-job/script.py",
        ))
    """

    argv_template: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.argv_template:
            raise ValueError("container command plan must contain at least one argument")
        if any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in self.argv_template):
            raise ValueError("container command arguments must be non-empty strings without NUL bytes")
        if any(_SECRET_ASSIGNMENT_RE.search(arg) for arg in self.argv_template):
            raise ValueError(
                "container command must reference credential environment names, not embed values"
            )
        runtime = Path(self.argv_template[0]).name
        if runtime not in _CONTAINER_RUNTIMES:
            raise ValueError(
                f"unsupported container runtime {runtime!r}; expected one of "
                f"{sorted(_CONTAINER_RUNTIMES)}"
            )

    def render(self, values: Mapping[str, str]) -> list[str]:
        allowed = {
            "job_dir",
            "script_path",
            "output_dir",
            "repo_root",
            "job_id",
            "method_id",
            "model_id",
        }
        rendered: list[str] = []
        for template in self.argv_template:
            result = template
            for key in allowed:
                result = result.replace("{" + key + "}", values[key])
            if "{" in result or "}" in result:
                raise ValueError(f"unknown or malformed container command placeholder in {template!r}")
            rendered.append(result)
        return rendered


@dataclass(frozen=True)
class ExecutionPolicy:
    """Select host execution or a caller-provided container launch plan."""

    mode: ExecutionMode = ExecutionMode.HOST
    container_command: ContainerCommandPlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ExecutionMode):
            object.__setattr__(self, "mode", ExecutionMode(self.mode))
        if self.mode is ExecutionMode.CONTAINER and self.container_command is None:
            raise ValueError("container execution requires a container command plan")
        if self.mode is ExecutionMode.HOST and self.container_command is not None:
            raise ValueError("host execution cannot include a container command plan")

    @classmethod
    def host(cls) -> "ExecutionPolicy":
        return cls(mode=ExecutionMode.HOST)

    @classmethod
    def containerized(cls, command: ContainerCommandPlan) -> "ExecutionPolicy":
        return cls(mode=ExecutionMode.CONTAINER, container_command=command)

    def command_argv(
        self,
        *,
        host_python: Path,
        script_path: Path,
        job_dir: Path,
        output_dir: str,
        repo_root: Path,
        job_id: str,
        method_id: str,
        model_id: str,
    ) -> list[str]:
        if self.mode is ExecutionMode.HOST:
            return [str(host_python), str(script_path)]
        assert self.container_command is not None
        return self.container_command.render(
            {
                "job_dir": str(job_dir),
                "script_path": str(script_path),
                "output_dir": output_dir,
                "repo_root": str(repo_root),
                "job_id": job_id,
                "method_id": method_id,
                "model_id": model_id,
            }
        )
