import os
import sys
import shutil
from pathlib import Path
import ruamel.yaml as yaml
from unittest import TestCase
from tempfile import TemporaryDirectory, NamedTemporaryFile
from moseq2_extract.tests.integration_tests.test_cli import write_fake_movie
from moseq2_extract.gui import update_progress, check_progress, generate_config_command, view_extraction, \
    generate_index_command, aggregate_extract_results_command, get_found_sessions, download_flip_command,\
    find_roi_command, sample_extract_command, extract_command, extract_found_sessions


class GUITests(TestCase):

    progress_vars = {'base_dir': './', 'config_file': 'TBD', 'index_file': 'TBD', 'train_data_dir': 'TBD',
                     'pca_dirname': 'TBD',
                     'scores_filename': 'TBD', 'scores_path': 'TBD', 'model_path': 'TBD', 'crowd_dir': 'TBD',
                     'plot_path': './plots/'}

    def test_update_progress(self):

        temp_prog = self.progress_vars

        with TemporaryDirectory() as tmp:
            progress_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml")

            with open(progress_path.name, 'w') as f:
                yaml.safe_dump(temp_prog, f)
            f.close()

            update_progress(progress_path.name, 'config_file', 1)

            # simulate opening file
            with open(progress_path.name, 'r') as f:
                test_progress = yaml.safe_load(f)
            f.close()

            assert test_progress != temp_prog, "dict was not saved correctly"


    def test_restore_progress_vars(self):
        temp_prog = self.progress_vars

        with TemporaryDirectory() as tmp:
            progress_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml")
            with open(progress_path.name, 'w') as f:
                yaml.safe_dump(temp_prog, f)
            f.close()

            # simulate opening file
            with open(progress_path.name, 'r') as f:
                progress1 = yaml.safe_load(f)
            f.close()

            assert progress1 == temp_prog, "dict was not returned correctly"

    def test_check_progress(self):

        # test file does not exist case
        with TemporaryDirectory() as tmp:
            progress_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml")
            outfile = Path(progress_path.name)

            # case: file does not exist
            if outfile.exists():
                os.remove(outfile)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(tmp, str(outfile))

            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 2, \
                "initial progress check failed"
            assert Path(progress_path.name).exists(), "progress yaml was not created"

            # simulate opening file
            with open(outfile, 'r') as f:
                progress1 = yaml.safe_load(f)
            f.close()

            for k,v in progress1.items():
                if k != 'base_dir' and k != 'plot_path':
                    assert v in self.progress_vars.values(), "read dict values to dont match simulated data"

            # now test case when file exists
            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()

            sys.stdin = open(stdin.name)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(tmp, str(outfile))

            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 2, \
                "progress retrieval from pre-existing file failed"

            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('N')
            f.close()

            sys.stdin = open(stdin.name)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(tmp, outfile)

            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 2, \
                "ignoring pre-existing progress file failed"

    def test_generate_config_command(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml").name)

            if config_path.is_file():
                os.remove(config_path)

            # file does not exist yet
            ret = generate_config_command(config_path)
            assert "success" in ret, "config file was not generated sucessfully"
            assert config_path.is_file(), "config file does not exist in specified path"

            # file exists
            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")

            # retain old version
            with open(stdin.name, 'w') as f:
                f.write('N')
            f.close()

            sys.stdin = open(stdin.name)

            ret = generate_config_command(config_path)
            assert "retained" in ret, "old config file was not retained"

            # overwrite old version
            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()

            sys.stdin = open(stdin.name)
            ret = generate_config_command(config_path)
            assert 'success' in ret, "overwriting failed"

    def test_view_extractions(self):
        extractions = ['1','2','3','4']

        with TemporaryDirectory() as tmp:
            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            # retain old version
            with open(stdin.name, 'w') as f:
                f.write('1,2,3')
            f.close()

            sys.stdin = open(stdin.name)

            ret = view_extraction(extractions, default=-1)
            assert len(ret) == 3, "function did not return the correct number of extractions to view"
            assert ret == ['1', '2', '3'], "function returned incorrect extractions to view"

    def test_generate_index_command(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(Path(tmp).parent, 'temp1')
            outfile = Path(input_dir, 'moseq2-index.yaml')

            if not input_dir.is_dir():
                input_dir.mkdir()

            # minimal test case - more use cases to come
            generate_index_command(input_dir, '', outfile, [], [])
            assert outfile.exists(), "index file was not generated correctly"

    def test_get_found_sessions(self):
        with TemporaryDirectory() as tmp:
            ft1 = NamedTemporaryFile(prefix=tmp+'/', suffix=".dat")
            ft2 = NamedTemporaryFile(prefix=tmp+'/', suffix=".mkv")
            ft3 = NamedTemporaryFile(prefix=tmp+'/', suffix=".avi")

            input_dir = Path(Path(tmp).parent, 'temp1')

            f1 = Path(input_dir, 'temp2/', Path(ft1.name).name)
            f2 = Path(input_dir, 'temp2/', Path(ft2.name).name)
            f3 = Path(input_dir, 'temp2/', Path(ft3.name).name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not f1.parent.exists():
                f1.parent.mkdir()
            else:
                for f in f1.parent.iterdir():
                    if f1.is_file():
                        f1.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)
                    elif f.is_file():
                        f.unlink()


            for i in [f1, f2, f3]:
                with open(i, 'w') as f:
                    f.write('Y')
                f.close()

            assert f1.is_file(), "temp file 1 was not created properly"
            assert f2.is_file(), "temp file 2 was not created properly"
            assert f3.is_file(), "temp file 3 was not created properly"

            data_dir, found_sessions = get_found_sessions(input_dir)
            assert(found_sessions == 3), "temp files were not successfully located"

            data_dir, found_sessions = get_found_sessions('')
            assert found_sessions == 0, "filesystem structure is incorrect. No sessions should be found."


    def test_download_flip_file_command(self):
        with TemporaryDirectory() as tmp:
            download_flip_command(tmp)
            assert True in [x.endswith('pkl') for x in os.listdir(tmp)], "flip file does not exist in correct directory"

    def test_find_roi_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml")
            configfile = Path(config_path.name)

            if configfile.is_file():
                os.remove(configfile)

            generate_config_command(configfile)

            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")

            # retain old version
            with open(stdin.name, 'w') as f:
                f.write('Q')
            f.close()

            sys.stdin = open(stdin.name)

            out = find_roi_command(tmp, configfile, select_session=True)
            assert (out == None), "roi function did not find any rois to extract"

            # writing a file to test following pipeline
            data_filepath = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".dat").name)

            input_dir = Path(Path(tmp).parent, 'temp1')
            data_path = Path(input_dir, 'temp2', data_filepath.name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not data_path.parent.is_dir():
                data_path.parent.mkdir()

            write_fake_movie(data_path)

            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('1')
            f.close()
            sys.stdin = open(stdin.name)

            images, filenames = find_roi_command(input_dir, configfile, select_session=True)
            assert (len(filenames) == 3), "incorrect number of rois were computed"
            assert (len(images) == 3), "incorrect number of rois images were computed"


    def test_sample_extract_command(self):

        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml")
            configfile = Path(config_path.name)

            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()
            sys.stdin = open(stdin.name)

            generate_config_command(configfile)

            # writing a file to test following pipeline
            data_filepath = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".dat").name)

            input_dir = Path(Path(tmp), 'temp1')
            data_path = Path(input_dir, 'temp2', data_filepath.name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not data_path.parent.is_dir():
                data_path.parent.mkdir()

            write_fake_movie(data_path)
            assert(data_path.is_file()), "fake movie was not written correctly"

            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('1')
            f.close()
            sys.stdin = open(stdin.name)

            output_dir = sample_extract_command(input_dir, configfile, 40, exts=['dat'], select_session=True)
            assert os.path.exists(output_dir), "sample_proc directory was not created"

    def test_extract_command(self):
        with TemporaryDirectory() as tmp:
            configfile = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml").name)

            if configfile.is_file():
                configfile.unlink()

            # writing a file to test following pipeline
            data_filepath = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".dat").name)

            input_dir = Path(Path(tmp), 'temp1')
            data_path = Path(input_dir, 'temp2', data_filepath.name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not data_path.parent.is_dir():
                data_path.parent.mkdir()

            write_fake_movie(data_path)
            assert(data_path.is_file()), "fake movie was not written correctly"

            generate_config_command(configfile)
            assert configfile.is_file()

            flip_file = Path('data/flip_classifier_k2_c57_10to13weeks.pkl')
            if not flip_file.is_file():
                download_flip_command('data/')

            assert flip_file.is_file()

            with open(configfile, 'r') as f:
                config_data = yaml.safe_load(f)

            config_data['flip_classifier'] = str(flip_file)

            with open(configfile, 'w') as f:
                yaml.safe_dump(config_data, f)

            stdin = NamedTemporaryFile(prefix=tmp+'/', suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()
            sys.stdin = open(stdin.name)

            ret = extract_command(data_path, None, configfile, skip=True)

            assert(Path(data_path.parent, 'proc').is_dir()), "proc directory was not created"
            assert(Path(data_path.parent, 'proc', 'done.txt').is_file()), "extraction did not finish"
            assert ('completed' in ret), "GUI command failed"
            os.remove(flip_file)

    def test_aggregate_results_command(self):
        with TemporaryDirectory() as tmp:
            ret = aggregate_extract_results_command(tmp, "", "aggregate_results")
            assert ret == os.path.join(tmp, 'moseq2-index.yaml'), "index file was not generated in correct directory"
            assert os.path.exists(os.path.join(tmp,'aggregate_results')), "aggregate results directory was not created"

    def test_extract_found_sessions(self):

        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp+'/', suffix=".yaml")
            configfile = Path(config_path.name)

            if configfile.is_file():
                configfile.unlink()

            generate_config_command(configfile)

            # writing a file to test following pipeline
            data_filepath = Path(NamedTemporaryFile(prefix=tmp+'/', suffix=".dat").name)

            input_dir = Path(Path(tmp).parent, 'temp1')
            data_path = Path(input_dir, 'temp2', data_filepath.name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not data_path.parent.is_dir():
                data_path.parent.mkdir()

            write_fake_movie(data_path)
            assert(data_path.is_file()), "fake movie was not written correctly"

            extract_found_sessions(data_path.parent, configfile, '.dat', skip_extracted=True)