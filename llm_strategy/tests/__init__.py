"""Unit and smoke tests for the LLM one-shot planner.

Pure unit tests run anywhere; ``test_smoke_e01`` instantiates the real
simulator and must be launched with the project's glibc-2.38 Python:

    cd /home/amax/ry/competition
    /home/amax/ry/competition/glibc-2.38/python3.11-glibc238 \
        -m unittest personal_train.llm_strategy.tests
"""
