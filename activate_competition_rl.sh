#!/usr/bin/env bash
# Source this file so a project-local prefix environment has a short prompt.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run: source /home/amax/ry/competition/personal_train/activate_competition_rl.sh" >&2
    exit 2
fi

_competition_rl_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_competition_rl_prefix="${_competition_rl_root}/.conda/competition-rl"

if [[ ! -x "${_competition_rl_prefix}/bin/python" ]]; then
    echo "Conda environment is missing: ${_competition_rl_prefix}" >&2
    unset _competition_rl_root _competition_rl_prefix
    return 1
fi

export CONDA_ENV_PROMPT='({name}) '
source /home/amax/miniconda3/etc/profile.d/conda.sh
conda activate "${_competition_rl_prefix}"

export PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR="${_competition_rl_root}/.cache/pip"

unset _competition_rl_root _competition_rl_prefix
