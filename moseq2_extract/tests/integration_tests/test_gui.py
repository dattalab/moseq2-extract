import os
import sys
import ruamel.yaml as yaml
from moseq2_extract.tests.integration_tests.test_cli import write_fake_movie
from unittest import TestCase
from tempfile import TemporaryDirectory, NamedTemporaryFile
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
            outfile = progress_path.name

            # case: file does not exist
            if os.path.exists(outfile):
                os.remove(outfile)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(os.path.dirname(progress_path.name), outfile)
            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 1

            assert os.path.exists(progress_path.name)

            # simulate opening file
            with open(outfile, 'r') as f:
                progress1 = yaml.safe_load(f)
            f.close()
            for k,v in progress1.items():
                if k != 'base_dir':
                    assert v in self.progress_vars.values()

            assert os.path.exists(progress_path.name)
            # now test case when file exists
            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()

            sys.stdin = open(stdin.name)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(os.path.dirname(progress_path.name), outfile)

            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 1

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('N')
            f.close()

            sys.stdin = open(stdin.name)

            config, index, tdd, pcadir, scores, model, score_path, cdir, pp = \
                check_progress(os.path.dirname(progress_path.name), outfile)

            assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 1

    def test_generate_config_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            outfile = config_path.name

            if os.path.exists(outfile):
                os.remove(outfile)

            # file does not exist yet
            ret = generate_config_command(outfile)
            assert "success" in ret
            assert os.path.exists(outfile)

            # file exists
            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # retain old version
            with open(stdin.name, 'w') as f:
                f.write('N')
            f.close()

            sys.stdin = open(stdin.name)

            ret = generate_config_command(outfile)
            assert "retained" in ret

            # overwrite old version
            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()

            sys.stdin = open(stdin.name)
            ret = generate_config_command(outfile)
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
            assert ret == ['1','2','3']

    def test_generate_index_command(self):
        with TemporaryDirectory() as tmp:
            input_dir = os.path.join(os.path.dirname(tmp), 'temp1')
            outfile = os.path.join(input_dir, 'moseq2-index.yaml')

            # minimal test case - more use cases to come
            generate_index_command(input_dir, '', outfile, [], [])
            assert os.path.exists(outfile)

    def test_get_found_sessions(self):
        with TemporaryDirectory() as tmp:
            input_dir = os.path.join(os.path.dirname(tmp), 'temp1')

            f1 = NamedTemporaryFile(prefix=tmp, suffix=".dat")
            f2 = NamedTemporaryFile(prefix=tmp, suffix=".mkv")
            f3 = NamedTemporaryFile(prefix=tmp, suffix=".avi")

            f1 = os.path.join(input_dir, os.path.dirname(f1.name), 'temp1', 'temp2',
                                     f1.name.split('/')[-1])
            f2 = os.path.join(input_dir, os.path.dirname(f2.name), 'temp1', 'temp2',
                              f2.name.split('/')[-1])
            f3 = os.path.join(input_dir, os.path.dirname(f3.name), 'temp1', 'temp2',
                              f3.name.split('/')[-1])

            if not os.path.exists(os.path.dirname(f1)):
                os.makedirs(os.path.dirname(f1))
            else:
                for f in os.listdir(os.path.dirname(f1)):
                    if os.path.isfile(os.path.join(os.path.dirname(f1), f)):
                        os.remove(os.path.join(os.path.dirname(f1), f))
                    elif os.path.isdir(f):
                        os.removedirs(os.path.join(os.path.dirname(f1), f))

            with open(f1, 'w') as f:
                f.write('Y')
            f.close()

            with open(f2, 'w') as f:
                f.write('Y')
            f.close()

            with open(f3, 'w') as f:
                f.write('Y')
            f.close()

            assert os.path.exists(f1)
            assert os.path.exists(f2)
            assert os.path.exists(f3)

            data_dir, found_sessions = get_found_sessions(input_dir)
            assert(found_sessions == 3)


    def test_download_flip_file_command(self):
        with TemporaryDirectory() as tmp:
            download_flip_command(tmp)
            assert True in [x.endswith('pkl') for x in os.listdir(tmp)]

    def test_find_roi_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            configfile = config_path.name

            if os.path.exists(configfile):
                os.remove(configfile)

            generate_config_command(configfile)

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")

            # retain old version
            with open(stdin.name, 'w') as f:
                f.write('Q')
            f.close()

            sys.stdin = open(stdin.name)

            out = find_roi_command(tmp, configfile)
            assert (out == None)

            # writing a file to test following pipeline
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".dat")
            input_dir = os.path.join(os.path.dirname(tmp), 'temp1')
            data_path = os.path.join(input_dir, os.path.dirname(data_path.name), 'temp1', 'temp2',
                                     data_path.name.split('/')[-1])
            if not os.path.exists(os.path.dirname(data_path)):
                os.makedirs(os.path.dirname(data_path))
            else:
                for f in os.listdir(os.path.dirname(data_path)):
                    if os.path.isfile(os.path.join(os.path.dirname(data_path), f)):
                        os.remove(os.path.join(os.path.dirname(data_path), f))
                    elif os.path.isdir(f):
                        os.removedirs(os.path.join(os.path.dirname(data_path), f))
            write_fake_movie(data_path)


            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('1')
            f.close()
            sys.stdin = open(stdin.name)

            images, filenames = find_roi_command(input_dir, configfile)
            assert (len(filenames) == 3)
            assert (len(images) == 3)


    def test_sample_extract_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            configfile = config_path.name

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('Y')
            f.close()
            sys.stdin = open(stdin.name)

            generate_config_command(configfile)

            # writing a file to test following pipeline
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".dat")

            input_dir = os.path.join(os.path.dirname(tmp), 'temp1')
            data_path = os.path.join(input_dir, os.path.dirname(data_path.name), 'temp1', 'temp2', data_path.name.split('/')[-1])
            if not os.path.exists(os.path.dirname(data_path)):
                os.makedirs(os.path.dirname(data_path))
            else:
                for f in os.listdir(os.path.dirname(data_path)):
                    if os.path.isfile(os.path.join(os.path.dirname(data_path), f)):
                        os.remove(os.path.join(os.path.dirname(data_path), f))
                    elif os.path.isdir(f):
                        os.removedirs(os.path.join(os.path.dirname(data_path), f))
            write_fake_movie(data_path)

            assert(os.path.exists(data_path))

            stdin = NamedTemporaryFile(prefix=tmp, suffix=".txt")
            # select test file
            with open(stdin.name, 'w') as f:
                f.write('1')
            f.close()
            sys.stdin = open(stdin.name)

            output_dir = sample_extract_command(input_dir, configfile, 40, exts=['dat'])
            assert os.path.exists(output_dir)

    def test_extract_command(self):
        with TemporaryDirectory() as tmp:
            config_path = NamedTemporaryFile(prefix=tmp, suffix=".yaml")
            configfile = config_path.name

            if os.path.exists(configfile):
                os.remove(configfile)

            generate_config_command(configfile)

            # writing a file to test following pipeline
            data_path = NamedTemporaryFile(prefix=tmp, suffix=".dat")

            input_dir = os.path.join(os.path.dirname(tmp), 'temp1')
            data_path = os.path.join(input_dir, os.path.dirname(data_path.name), 'temp1', 'temp2',
                                     data_path.name.split('/')[-1])
            if not os.path.exists(os.path.dirname(data_path)):
                os.makedirs(os.path.dirname(data_path))
            else:
                for f in os.listdir(os.path.dirname(data_path)):
                    if os.path.isfile(os.path.join(os.path.dirname(data_path), f)):
                        os.remove(os.path.join(os.path.dirname(data_path), f))
                    elif os.path.isdir(f):
                        os.removedirs(os.path.join(os.path.dirname(data_path), f))

            write_fake_movie(data_path)

            assert(os.path.exists(data_path))
            ret = extract_command(data_path, None, configfile, skip=True)
            assert('proc' in os.listdir(os.path.dirname(data_path)))
            assert ('done.txt' in os.listdir(os.path.join(os.path.dirname(data_path), 'proc')))
            assert ('completed' in ret)

    def test_aggregate_results_command(self):
        with TemporaryDirectory() as tmp:
            ret = aggregate_extract_results_command(tmp, "", "aggregate_results")
            assert ret == os.path.join(tmp, 'moseq2-index.yaml')
            assert os.path.exists(os.path.join(tmp,'aggregate_results'))

    def test_extract_found_sessions(self):
        print('not implemented: will test individual components first.')