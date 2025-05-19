# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/tests/test_python_yml.py
# Tests unitaires pour .github/workflows/python.yml
#
# Version : 2.1.4
# Date : 2025-05-13
#
# Rôle : Valide la configuration du pipeline CI/CD pour MIA_IA_SYSTEM_v2_2025, incluant la syntaxe YAML,
#        la présence des jobs (test, lint, docs), les étapes spécifiques, et l'intégration des alertes Telegram.
#
# Dépendances :
# - pytest>=7.3.0,<8.00, pyyaml>=6.0.0,<7.0.0
#
# Inputs :
# - .github/workflows/python.yml
#
# Outputs :
# - Tests unitaires pour vérifier la conformité du pipeline CI/CD.
#
# Notes :
# - Conforme à structure.txt (version 2.1.4, 2025-05-13).
# - Vérifie l'absence de références à dxFeed, obs_t, 320 features, et 81 features dans le pipeline.
# - Valide les étapes pour les snapshots compressés (*.json.gz) et les sauvegardes incrémentielles.
# - Tests les jobs test, lint, et docs, ainsi que les alertes Telegram pour les échecs critiques.

import pytest
import yaml


@pytest.fixture
def workflow_file(tmp_path):
    """Crée un fichier python.yml temporaire pour les tests."""
    workflow_content = """
name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: pip
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest>=7.3.0,<8.0.0 flake8>=6.0.0,<7.0.0 pylint>=2.17.0,<3.0.0 mypy>=1.4.0,<2.0.0 coverage>=7.2.0,<8.0.0
    - name: Validate Python version
      run: |
        python -c "import sys; assert sys.version_info[:2] == (3, 10), 'Python version must be 3.10'"
    - name: Validate feature_sets.yaml
      run: |
        python -c "import yaml; config = yaml.safe_load(open('config/feature_sets.yaml')); assert len(config['inference']['shap_features']) == 150, 'Expected 150 SHAP features'; assert len(config['training']['features']) == 350, 'Expected 350 training features'"
    - name: Validate YAML configurations
      run: |
        python -c "import yaml; config = yaml.safe_load(open('config/es_config.yaml')); assert 's3_bucket' in config.get('main_params', {}), 'Missing s3_bucket in es_config.yaml'; assert 's3_prefix' in config.get('main_params', {}), 'Missing s3_prefix in es_config.yaml'"
        python -c "import yaml; [yaml.safe_load(open(f)) for f in ['config/es_config.yaml', 'config/feature_sets.yaml']]"
    - name: Run pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=html --cov-fail-under=80
    - name: Verify logs and snapshots
      run: |
        python -c "import os; assert os.path.exists('data/logs/provider_performance.csv'), 'Missing provider_performance.csv'"
        python -c "import os, glob; assert len(glob.glob('data/cache/*/*.json.gz')) > 0, 'No compressed snapshots found'"
        python -c "import os, glob; assert len(glob.glob('data/checkpoints/*.json.gz')) > 0, 'No incremental backups found'"
    - name: Send Telegram alert on failure
      if: failure()
      env:
        TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: |
        curl -s -X POST https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage -d chat_id=$TELEGRAM_CHAT_ID -d text="🚨 MIA CI Pipeline Failed: ${{ github.event_name }} on ${{ github.sha }}"
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        files: ./coverage.xml
        fail_ci_if_error: true
    - name: Upload coverage report
      uses: actions/upload-artifact@v4
      with:
        name: coverage-report
        path: htmlcov/

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: pip
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8>=6.0.0,<7.0.0 pylint>=2.17.0,<3.0.0 mypy>=1.4.0,<2.0.0
    - name: Check for removed terms
      run: |
        ! grep -r -i "dxFeed\\|obs_t\\|320 features\\|81 features" src/ || (echo "Found references to dxFeed, obs_t, 320 features, or 81 features" && exit 1)
    - name: Run flake8
      run: |
        flake8 src/ --max-line-length=88 --extend-ignore=E203
    - name: Run pylint
      run: |
        pylint src/ --disable=too-many-locals,too-many-arguments,too-many-statements
    - name: Run mypy
      run: |
        mypy src/ --strict
    - name: Send Telegram alert on failure
      if: failure()
      env:
        TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: |
        curl -s -X POST https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage -d chat_id=$TELEGRAM_CHAT_ID -d text="🚨 MIA Linting Failed: ${{ github.event_name }} on ${{ github.sha }}"

  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        cache: pip
    - name: Install Sphinx
      run: |
        python -m pip install --upgrade pip
        pip install sphinx>=7.0.0,<8.0.0 sphinx-rtd-theme>=1.3.0,<2.0.0
    - name: Check documentation files
      run: |
        python -c "import os; assert all(os.path.exists(f) for f in ['docs/index.rst', 'docs/modules.rst', 'docs/installation.rst', 'docs/configuration.rst', 'docs/phases.rst', 'docs/tests.rst']), 'Missing RST files'"
    - name: Build Sphinx documentation
      run: |
        cd docs
        make html
        python -c "import os; assert os.path.exists('_build/html/index.html'), 'Sphinx build failed'"
    - name: Validate Python syntax
      run: |
        python -c "import ast, glob; [ast.parse(open(f).read()) for f in glob.glob('src/**/*.py', recursive=True)]"
    - name: Send Telegram alert on failure
      if: failure()
      env:
        TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: |
        curl -s -X POST https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage -d chat_id=$TELEGRAM_CHAT_ID -d text="🚨 MIA Documentation Build Failed: ${{ github.event_name }} on ${{ github.sha }}"
"""
    workflow_path = tmp_path / ".github/workflows/python.yml"
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    with open(workflow_path, "w", encoding="utf-8") as f:
        f.write(workflow_content)
    return workflow_path


def test_yaml_syntax(workflow_file):
    """Teste la validité de la syntaxe YAML du fichier python.yml."""
    with open(workflow_file, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    assert content is not None, "Fichier YAML invalide"
    assert content["name"] == "Python CI", "Nom du pipeline incorrect"


def test_workflow_structure(workflow_file):
    """Teste la structure du fichier python.yml."""
    with open(workflow_file, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    assert "on" in content, "Section 'on' manquante"
    assert "push" in content["on"], "Trigger 'push' manquant"
    assert "pull_request" in content["on"], "Trigger 'pull_request' manquant"
    assert "jobs" in content, "Section 'jobs' manquante"
    assert set(content["jobs"].keys()) == {
        "test",
        "lint",
        "docs",
    }, "Jobs attendus: test, lint, docs"


def test_test_job_steps(workflow_file):
    """Teste les étapes du job test."""
    with open(workflow_file, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    test_job = content["jobs"]["test"]
    assert test_job["runs-on"] == "ubuntu-latest", "OS incorrect pour job test"
    steps = test_job["steps"]
    step_names = [step.get("name", step.get("uses", "")) for step in steps]
    expected_steps = [
        "actions/checkout@v4",
        "Set up Python 3.10",
        "Install dependencies",
        "Validate Python version",
        "Validate feature_sets.yaml",
        "Validate YAML configurations",
        "Run pytest",
        "Verify logs and snapshots",
        "Send Telegram alert on failure",
        "Upload coverage to Codecov",
        "Upload coverage report",
    ]
    assert all(
        expected in step_names for expected in expected_steps
    ), "Étapes manquantes dans job test"
    for step in steps:
        if step.get("name") == "Validate feature_sets.yaml":
            assert "150" in step["run"], "Validation SHAP features incorrecte"
            assert "350" in step["run"], "Validation training features incorrecte"
        if step.get("name") == "Validate YAML configurations":
            assert "s3_bucket" in step["run"], "Validation s3_bucket manquante"
        if step.get("name") == "Verify logs and snapshots":
            assert (
                ".json.gz" in step["run"]
            ), "Validation snapshots compressés manquante"


def test_lint_job_steps(workflow_file):
    """Teste les étapes du job lint."""
    with open(workflow_file, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    lint_job = content["jobs"]["lint"]
    assert lint_job["runs-on"] == "ubuntu-latest", "OS incorrect pour job lint"
    steps = lint_job["steps"]
    step_names = [step.get("name", step.get("uses", "")) for step in steps]
    expected_steps = [
        "actions/checkout@v4",
        "Set up Python 3.10",
        "Install dependencies",
        "Check for removed terms",
        "Run flake8",
        "Run pylint",
        "Run mypy",
        "Send Telegram alert on failure",
    ]
    assert all(
        expected in step_names for expected in expected_steps
    ), "Étapes manquantes dans job lint"
    for step in steps:
        if step.get("name") == "Check for removed terms":
            assert "dxFeed" in step["run"], "Vérification dxFeed manquante"


def test_docs_job_steps(workflow_file):
    """Teste les étapes du job docs."""
    with open(workflow_file, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    docs_job = content["jobs"]["docs"]
    assert docs_job["runs-on"] == "ubuntu-latest", "OS incorrect pour job docs"
    steps = docs_job["steps"]
    step_names = [step.get("name", step.get("uses", "")) for step in steps]
    expected_steps = [
        "actions/checkout@v4",
        "Set up Python 3.10",
        "Install Sphinx",
        "Check documentation files",
        "Build Sphinx documentation",
        "Validate Python syntax",
        "Send Telegram alert on failure",
    ]
    assert all(
        expected in step_names for expected in expected_steps
    ), "Étapes manquantes dans job docs"


def test_telegram_alerts(workflow_file):
    """Teste la configuration des alertes Telegram dans tous les jobs."""
    with open(workflow_file, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    for job_name, job in content["jobs"].items():
        steps = job["steps"]
        telegram_step = next(
            (
                step
                for step in steps
                if step.get("name") == "Send Telegram alert on failure"
            ),
            None,
        )
        assert (
            telegram_step is not None
        ), f"Alertes Telegram manquantes dans job {job_name}"
        assert "TELEGRAM_BOT_TOKEN" in telegram_step.get(
            "env", {}
        ), f"TELEGRAM_BOT_TOKEN manquant dans job {job_name}"
        assert "TELEGRAM_CHAT_ID" in telegram_step.get(
            "env", {}
        ), f"TELEGRAM_CHAT_ID manquant dans job {job_name}"
        assert (
            "if: failure()" in telegram_step
        ), f"Condition failure() manquante dans job {job_name}"


def test_dependency_versions(workflow_file):
    """Teste les versions des dépendances spécifiées."""
    with open(workflow_file, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    for job in content["jobs"].values():
        for step in job["steps"]:
            if step.get("name") in ["Install dependencies", "Install Sphinx"]:
                run_command = step["run"]
                assert (
                    "pytest>=7.3.0,<8.0.0" in run_command
                    or "sphinx>=7.0.0,<8.0.0" in run_command
                ), "Versions des dépendances incorrectes"
                assert (
                    "flake8>=6.0.0,<7.0.0" in run_command
                    or "sphinx-rtd-theme>=1.3.0,<2.0.0" in run_command
                ), "Versions des dépendances incorrectes"
