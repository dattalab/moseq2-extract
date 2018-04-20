import pytest
from click.testing import CliRunner
from moseq2_extract.cli import find_roi, extract, download_flip_file, generate_config


def test_extract():

    runner = CliRunner()
    result = runner.invoke(extract, ['--help'])
    assert(result.exit_code == 0)


def test_find_roi():

    runner = CliRunner()
    result = runner.invoke(find_roi, ['--help'])
    assert(result.exit_code == 0)


def test_download_flip_file():

    runner = CliRunner()
    result = runner.invoke(download_flip_file, ['--help'])
    assert(result.exit_code == 0)


def test_generate_config():

    runner = CliRunner()
    result = runner.invoke(generate_config, ['--help'])
    assert(result.exit_code == 0)
