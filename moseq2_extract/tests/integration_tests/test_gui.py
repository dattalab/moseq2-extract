import os
import sys
import shutil
from pathlib import Path
import ruamel.yaml as yaml
from unittest import TestCase
from tempfile import TemporaryDirectory, NamedTemporaryFile
from moseq2_extract.tests.integration_tests.test_cli import write_fake_movie
from moseq2_extract.gui import check_progress, generate_config_command, view_extraction, \
    generate_index_command, aggregate_extract_results_command, get_found_sessions, download_flip_command,\
    find_roi_command, sample_extract_command, extract_command


class GUITests(TestCase):

    progress_vars = {'base_dir': './', 'config_file': 'TBD', 'index_file': 'TBD', 'train_data_dir': 'TBD',
                     'pca_dirname': 'TBD',
                     'scores_filename': 'TBD', 'scores_path': 'TBD', 'model_path': 'TBD', 'crowd_dir': 'TBD',
                     'plot_path': 'TBD'}

    def test_update_progress(self):

        temp_prog = self.progress_vars

        with TemporaryDirectory() as tmp:
            progress_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            with open(progress_path.name, 'w') as f:
                yaml.safe_dump(temp_prog, f)
            f.close()

            # simulate opening file
            with open(progress_path.name, 'r') as f:
                progress = yaml.safe_load(f)
            f.close()

            # simulate update for all supported keys
            for key in temp_prog.keys():
                temp_prog[key] = 0
                progress[key] = 0

            assert all(progress.values()) == 0

            # simulate write
            with open(progress_path.name, 'w') as f:
                yaml.safe_dump(progress, f)
            f.close()

            # simulate opening file
            with open(progress_path.name, 'r') as f:
                progress1= yaml.safe_load(f)
            f.close()

            assert progress1 == temp_prog


    def test_restore_progress_vars(self):
        temp_prog = self.progress_vars

        with TemporaryDirectory() as tmp:
            progress_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            with open(progress_path.name, 'w') as f:
                yaml.safe_dump(temp_prog, f)
            f.close()

            # simulate opening file
            with open(progress_path.name, 'r') as f:
                progress1 = yaml.safe_load(f)
            f.close()

            assert progress1 == temp_prog

    def test_check_progress(self):

        # test file does not exist case
        with TemporaryDirectory() as tmp:
            progress_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            outfile = Path(progress_path.name)

            # case: file does not exist
            if outfile.exists():
                os.remove(outfile)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(tmp, str(outfile))

            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 1
            assert Path(progress_path.name).exists()

            # simulate opening file
            with open(outfile, 'r') as f:
                progress1 = yaml.safe_load(f)
            f.close()
            for k,v in progress1.items():
                if k != 'base_dir':
                    assert v in self.progress_vars.values()

            assert Path(progress_path.name).exists()

            # now test case when file exists
            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()

            sys.stdin = open(stdin.name)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(tmp, str(outfile))

            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 1

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('N')
            f.close()

            sys.stdin = open(stdin.name)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(tmp, str(outfile))

            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 1

    def test_generate_config_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            outfile = Path(config_path.name)

            if outfile.exists():
                os.remove(outfile)

            # file does not exist yet
            ret = generate_config_command(str(outfile))
            assert "success" in ret
            assert outfile.exists()

            # file exists
            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # retain old version
            with open(stdin.name, 'w') as f:
                f.write('N')
            f.close()

            sys.stdin = open(stdin.name)

            ret = generate_config_command(str(outfile))
            assert "retained" in ret

            # overwrite old version
            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()

            sys.stdin = open(stdin.name)
            ret = generate_config_command(str(outfile))
            assert 'success' in ret

    def test_view_extractions(self):
        extractions = ['1','2','3','4']

        with TemporaryDirectory() as tmp:
            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # retain old version
            with open(stdin.name, 'w') as f:
                f.write('1,2,3')
            f.close()

            sys.stdin = open(stdin.name)

            ret = view_extraction(extractions)
            assert len(ret) == 3
            assert ret == ['1', '2', '3']

    def test_generate_index_command(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp).resolve().parent.joinpath('temp1')
            outfile = input_dir.joinpath('moseq2-index.yaml')

            if not input_dir.is_dir():
                input_dir.mkdir()

            # minimal test case - more use cases to come
            generate_index_command(str(input_dir), '', str(outfile), [], [])
            assert outfile.exists()

    def test_get_found_sessions(self):
        with TemporaryDirectory() as tmp:
            ft1 = NamedTemporaryFile(prefix=tmp, suffix=".dat")
            ft2 = NamedTemporaryFile(prefix=tmp, suffix=".mkv")
            ft3 = NamedTemporaryFile(prefix=tmp, suffix=".avi")

            input_dir = Path(tmp).resolve().parent.joinpath('temp1')

            f1 = input_dir.joinpath('temp2/', Path(ft1.name).name)
            f2 = input_dir.joinpath('temp2/', Path(ft2.name).name)
            f3 = input_dir.joinpath('temp2/', Path(ft3.name).name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not f1.parent.exists():
                f1.parent.mkdir()
            else:
                for f in f1.parent.iterdir():
                    if f1.resolve().is_file():
                        os.remove(f1.resolve())
                    elif f.is_dir():
                        shutil.rmtree(str(f))
                    elif f.is_file():
                        os.remove(str(f))

            with open(f1, 'w') as f:
                f.write('Y')
            f.close()

            with open(f2, 'w') as f:
                f.write('Y')
            f.close()

            with open(f3, 'w') as f:
                f.write('Y')
            f.close()

            assert f1.is_file()
            assert f2.is_file()
            assert f3.is_file()

            data_dir, found_sessions = get_found_sessions(str(input_dir))
            assert(found_sessions == 3)


    def test_download_flip_file_command(self):
        with TemporaryDirectory() as tmp:
            download_flip_command(tmp)
            assert True in [x.endswith('pkl') for x in os.listdir(tmp)]

    def test_find_roi_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            configfile = Path(config_path.name)

            if configfile.is_file():
                os.remove(configfile)

            generate_config_command(configfile)

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")

            # retain old version
            with open(stdin.name, 'w') as f:
                f.write('Q')
            f.close()

            sys.stdin = open(stdin.name)

            out = find_roi_command(tmp, str(configfile))
            assert (out == None)

            # writing a file to test following pipeline
            data_filepath = NamedTemporaryFile(prefix=tmp, suffix=".dat")

            input_dir = Path(tmp).resolve().parent.joinpath('temp1')
            data_path = input_dir.joinpath('temp2', Path(data_filepath.name).name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not data_path.parent.is_dir():
                data_path.parent.mkdir()
            else:
                for f in data_path.parent.iterdir():
                    if f.is_file():
                        os.remove(f.resolve())
                    elif f.is_dir():
                        shutil.rmtree(str(f))
            write_fake_movie(data_path)

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('1')
            f.close()
            sys.stdin = open(stdin.name)

            images, filenames = find_roi_command(str(input_dir), str(configfile))
            assert (len(filenames) == 3)
            assert (len(images) == 3)


    def test_sample_extract_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            configfile = Path(config_path.name)

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()
            sys.stdin = open(stdin.name)

            generate_config_command(str(configfile))

            # writing a file to test following pipeline
            data_filepath = NamedTemporaryFile(prefix=tmp, suffix=".dat")

            input_dir = Path(tmp).resolve().parent.joinpath('temp1')
            data_path = input_dir.joinpath('temp2', Path(data_filepath.name).name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not data_path.parent.is_dir():
                data_path.parent.mkdir()
            else:
                for f in data_path.parent.iterdir():
                    print(f)
                    if f.is_file():
                        os.remove(f.resolve())
                    elif f.is_dir():
                        shutil.rmtree(str(f))

            write_fake_movie(data_path)
            assert(data_path.is_file())

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('1')
            f.close()
            sys.stdin = open(stdin.name)

            output_dir = sample_extract_command(str(input_dir), str(configfile), 40, exts=['dat'])
            assert os.path.exists(output_dir)

    def test_extract_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            configfile = Path(config_path.name)

            if configfile.is_file():
                configfile.unlink()

            generate_config_command(str(configfile))

            # writing a file to test following pipeline
            data_filepath = NamedTemporaryFile(prefix=tmp, suffix=".dat")

            input_dir = Path(tmp).resolve().parent.joinpath('temp1')
            data_path = input_dir.joinpath('temp2', Path(data_filepath.name).name)

            if not input_dir.is_dir():
                input_dir.mkdir()

            if not data_path.parent.is_dir():
                data_path.parent.mkdir()
            else:
                for f in data_path.parent.iterdir():
                    print(f)
                    if f.is_file():
                        os.remove(f.resolve())
                    elif f.is_dir():
                        shutil.rmtree(str(f))

            write_fake_movie(data_path)
            assert(data_path.is_file())

            ret = extract_command(str(data_path), None, str(configfile), skip=True)

            assert(data_path.parent.joinpath('proc').is_dir())
            assert(data_path.parent.joinpath('proc', 'done.txt').is_file())
            assert ('completed' in ret)

    def test_aggregate_results_command(self):
        with TemporaryDirectory() as tmp:
            ret = aggregate_extract_results_command(tmp, "", "aggregate_results")
            assert ret == os.path.join(tmp, 'moseq2-index.yaml')
            assert os.path.exists(os.path.join(tmp,'aggregate_results'))

    def test_extract_found_sessions(self):
        print('not implemented: will test individual components first.')