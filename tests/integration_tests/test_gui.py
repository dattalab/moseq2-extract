import os
import sys
import shutil
import ruamel.yaml as yaml
from unittest import TestCase
from .test_cli import write_fake_movie
from moseq2_extract.helpers.wrappers import copy_h5_metadata_to_yaml_wrapper
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

        progress_path = 'data/test_progress.yaml'

        with open(progress_path, 'w') as f:
            yaml.safe_dump(temp_prog, f)

        update_progress(progress_path, 'config_file', 1)

        # simulate opening file
        with open(progress_path, 'r') as f:
            test_progress = yaml.safe_load(f)

        assert test_progress != temp_prog, "dict was not saved correctly"
        os.remove(progress_path)

    def test_restore_progress_vars(self):
        temp_prog = self.progress_vars

        progress_path = 'data/test_restore_progress.yaml'
        with open(progress_path, 'w') as f:
            yaml.safe_dump(temp_prog, f)

        # simulate opening file
        with open(progress_path, 'r') as f:
            progress1 = yaml.safe_load(f)

        assert progress1 == temp_prog, "dict was not returned correctly"
        os.remove(progress_path)

    def test_check_progress(self):

        # test file does not exist case
        input_dir = 'data/'
        progress_path = 'data/test_check_progress.yaml'

        # case: file does not exist
        if os.path.isfile(progress_path):
            os.remove(progress_path)

        config, index, tdd, pcadir, scores, model, score_path, cdir, pp = check_progress(input_dir, progress_path)

        assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 2, \
            "initial progress check failed"
        assert os.path.isfile(os.path.join(progress_path)), "progress yaml was not created"

        # simulate opening file
        with open(progress_path, 'r') as f:
            progress1 = yaml.safe_load(f)

        for k,v in progress1.items():
            if k != 'base_dir' and k != 'plot_path':
                assert v in self.progress_vars.values(), "read dict values to dont match simulated data"

        # now test case when file exists
        stdin = 'data/tmp_stdin.txt'
        with open(stdin, 'w') as f:
            f.write('Y')

        sys.stdin = open(stdin)

        config, index, tdd, pcadir, scores, model, score_path, cdir, pp = check_progress(input_dir, progress_path)

        assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 2, \
            "progress retrieval from pre-existing file failed"

        with open(stdin, 'w') as f:
            f.write('N')

        sys.stdin = open(stdin)

        config, index, tdd, pcadir, scores, model, score_path, cdir, pp = check_progress(input_dir, progress_path)

        assert len(set([config, index, tdd, pcadir, scores, model, score_path, cdir, pp])) == 2, \
            "ignoring pre-existing progress file failed"
        os.remove(stdin)
        os.remove(progress_path)

    def test_generate_config_command(self):

        config_path = 'data/test_config.yaml'

        if os.path.isfile(config_path):
            os.remove(config_path)

        # file does not exist yet
        ret = generate_config_command(config_path)
        assert "success" in ret, "config file was not generated sucessfully"
        assert os.path.isfile(config_path), "config file does not exist in specified path"

        # file exists
        stdin = 'data/tmp_stdin.txt'

        # retain old version
        with open(stdin, 'w') as f:
            f.write('N')

        sys.stdin = open(stdin)

        ret = generate_config_command(config_path)
        assert "retained" in ret, "old config file was not retained"

        # overwrite old version
        with open(stdin, 'w') as f:
            f.write('Y')

        sys.stdin = open(stdin)
        ret = generate_config_command(config_path)
        assert 'success' in ret, "overwriting failed"
        os.remove(config_path)
        os.remove(stdin)

    def test_view_extractions(self):
        extractions = ['1','2','3','4']

        stdin = 'data/tmp_stdin.txt'
        # retain old version
        with open(stdin, 'w') as f:
            f.write('1,2,3')

        sys.stdin = open(stdin)

        ret = view_extraction(extractions, default=-1)
        assert len(ret) == 3, "function did not return the correct number of extractions to view"
        assert ret == ['1', '2', '3'], "function returned incorrect extractions to view"
        os.remove(stdin)

    def test_generate_index_command(self):

        input_dir = 'data/'
        outfile = os.path.join(input_dir, 'moseq2-index.yaml')

        # minimal test case - more use cases to come
        generate_index_command(input_dir, '', outfile, [], [])
        assert os.path.isfile(outfile), "index file was not generated correctly"
        os.remove(outfile)

    def test_get_found_sessions(self):

        ft1 = 'tmp_depth.dat'
        ft2 = 'tmp_depth.mkv'
        ft3 = 'tmp_depth.avi'

        input_dir = 'data/'
        temp_outdir = os.path.join(input_dir, 'temp')

        f1 = os.path.join(temp_outdir, ft1)
        f2 = os.path.join(temp_outdir, ft2)
        f3 = os.path.join(temp_outdir, ft3)

        if not os.path.isdir(temp_outdir):
            os.makedirs(temp_outdir)

        for i in [f1, f2, f3]:
            with open(i, 'w') as f:
                f.write('Y')

        assert os.path.isfile(f1), "temp file 1 was not created properly"
        assert os.path.isfile(f2), "temp file 2 was not created properly"
        assert os.path.isfile(f3), "temp file 3 was not created properly"

        data_dir, found_sessions = get_found_sessions(input_dir)
        assert(found_sessions == 3), "temp files were not successfully located"

        data_dir, found_sessions = get_found_sessions('')
        assert found_sessions == 0, "filesystem structure is incorrect. No sessions should be found."
        shutil.rmtree(temp_outdir)

    def test_download_flip_file_command(self):
        test_outdir = 'data/flip/'
        download_flip_command(test_outdir)
        assert True in [x.endswith('pkl') for x in os.listdir(test_outdir)], "flip file does not exist in correct directory"
        shutil.rmtree(test_outdir)

    def test_find_roi_command(self):

        config_path = 'data/test_roi_config.yaml'

        if os.path.isfile(config_path):
            os.remove(config_path)

        generate_config_command(config_path)

        out = find_roi_command('data/', config_path, select_session=True)
        assert (out == None), "roi function did not find any rois to extract"

        # writing a file to test following pipeline
        input_dir = 'data/'
        data_path = 'data/test_session/'
        data_filepath = os.path.join(data_path, 'test_roi_depth.dat')

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        write_fake_movie(data_filepath)
        assert os.path.isfile(data_filepath)

        stdin = 'data/stdin.txt'
        # select test file
        with open(stdin, 'w') as f:
            f.write('1')

        sys.stdin = open(stdin)

        images, filenames = find_roi_command(input_dir, config_path, select_session=True)
        assert (len(filenames) == 3), "incorrect number of rois were computed"
        assert (len(images) == 3), "incorrect number of rois images were computed"
        shutil.rmtree(data_path)
        os.remove(stdin)
        os.remove(config_path)

    def test_sample_extract_command(self):

        config_path = 'data/test_sample_ex_config.yaml'
        generate_config_command(config_path)

        # writing a file to test following pipeline
        input_dir = 'data/'
        data_path = 'data/test_session/'
        data_filepath = os.path.join(data_path, 'test_sample_extract_depth.dat')

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        write_fake_movie(data_filepath)
        assert(os.path.isfile(data_filepath)), "fake movie was not written correctly"

        stdin = 'data/stdin.txt'

        # select test file
        with open(stdin, 'w') as f:
            f.write('1')

        sys.stdin = open(stdin)

        output_dir = sample_extract_command(input_dir, config_path, 40, exts=['dat'], select_session=True)
        assert os.path.exists(output_dir), "sample_proc directory was not created"
        shutil.rmtree(data_path)
        os.remove(stdin)
        os.remove(config_path)

    def test_extract_command(self):
        configfile = 'data/test_ex_config.yaml'

        # writing a file to test following pipeline
        data_path = 'data/test_extract/'
        data_filepath = os.path.join(data_path, 'test_extract_depth.dat')
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

        write_fake_movie(data_filepath)
        assert(os.path.isfile(data_filepath)), "fake movie was not written correctly"

        if os.path.isfile(configfile):
            os.remove(configfile)

        generate_config_command(configfile)
        assert os.path.isfile(configfile)

        flip_file = 'data/flip/flip_classifier_k2_c57_10to13weeks.pkl'
        if not os.path.isfile(flip_file):
            download_flip_command('data/flip/')

        assert os.path.isfile(flip_file), 'flip file was not correctly downloaded'

        with open(configfile, 'r') as f:
            config_data = yaml.safe_load(f)

        config_data['flip_classifier'] = flip_file
        config_data['use_plane_bground'] = True

        with open(configfile, 'w') as f:
            yaml.safe_dump(config_data, f)

        stdin = 'data/stdin.txt'
        with open(stdin, 'w') as f:
            f.write('Y')

        sys.stdin = open(stdin)

        ret = extract_command(data_filepath, None, configfile, skip=True)

        out_dir = 'data/test_extract/proc/'

        assert(os.path.isdir(out_dir)), "proc directory was not created"
        assert(os.path.isfile(os.path.join(out_dir, 'done.txt'))), "extraction did not finish"
        assert ('completed' in ret), "GUI command failed"
        shutil.rmtree('data/flip/')
        shutil.rmtree(data_path)
        os.remove(configfile)
        os.remove(stdin)

    def test_aggregate_results_command(self):
        input_dir = 'data/'
        ret = aggregate_extract_results_command(input_dir, "", "aggregate_results")

        assert ret == os.path.join(input_dir, 'moseq2-index.yaml'), "index file was not generated in correct directory"
        assert os.path.isfile(os.path.join(input_dir, 'moseq2-index.yaml'))
        assert os.path.isdir(os.path.join(input_dir,'aggregate_results')), "aggregate results directory was not created"

        shutil.rmtree(os.path.join(input_dir,'aggregate_results'))
        os.remove(os.path.join(input_dir, 'moseq2-index.yaml'))

    def test_extract_found_sessions(self):

        config_path = 'data/test_config.yaml'
        generate_config_command(config_path)

        # writing a file to test following pipeline
        data_filepath = 'data/test_extract_found.dat'

        write_fake_movie(data_filepath)
        assert(os.path.isfile(data_filepath)), "fake movie was not written correctly"

        extract_found_sessions(data_filepath, config_path, '.dat', skip_extracted=True)
        os.remove(data_filepath)
        os.remove(config_path)

    def test_copy_h5_metadata_to_yaml(self):
        input_dir = 'data/proc/'
        h5_metadata_path = '/metadata/acquisition/'

        # Functionality check
        copy_h5_metadata_to_yaml_wrapper(input_dir, h5_metadata_path)