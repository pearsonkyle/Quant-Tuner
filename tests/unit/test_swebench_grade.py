

# --- V1 / V2 classification ------------------------------------------------------
# Regression guard for a silent, expensive failure: keying is_v2_instance on
# `log_parser` alone classified the whole V1 Python holdout as V2, pointed the agent at
# an empty /<repo-name> Docker created for it, and every instance returned
# "fatal: not a git repository" as its patch -> a reported 100% patch rate on nothing.

V1_ROW = {
    "instance_id": "Azure__azure-cli-2214", "repo": "Azure/azure-cli",
    "image_name": "swerebench/sweb.eval.x86_64.azure_1776_azure-cli-2214",
    "install_config": {"log_parser": "parse_log_pytest", "env_yml_path": "env.yml",
                       "reqs_path": "r.txt", "packages": "x", "python": "3.9",
                       "no_use_env": False, "test_cmd": "pytest"},
}
V2_ROW = {
    "instance_id": "pactus-project__pactus-669", "repo": "pactus-project/pactus",
    "image_name": "docker.io/swerebenchv2/pactus-project-pactus:669-abc",
    "install_config": {"log_parser": "parse_log_gotest", "test_cmd": "go test ./..."},
}


def test_v1_python_row_is_not_v2_despite_naming_a_log_parser():
    from quant_tuner.eval.swebench_grade import is_v2_instance, workdir_for

    assert is_v2_instance(V1_ROW) is False
    assert workdir_for(V1_ROW) == "/testbed"


def test_v2_row_is_detected_and_uses_the_repo_workdir():
    from quant_tuner.eval.swebench_grade import is_v2_instance, workdir_for

    assert is_v2_instance(V2_ROW) is True
    assert workdir_for(V2_ROW) == "/pactus"


def test_install_config_shape_decides_when_the_image_is_absent():
    from quant_tuner.eval.swebench_grade import is_v2_instance

    v1 = {**V1_ROW}
    v1.pop("image_name")
    v2 = {**V2_ROW}
    v2.pop("image_name")
    assert is_v2_instance(v1) is False   # conda keys -> V1
    assert is_v2_instance(v2) is True    # parser, no conda keys -> V2
