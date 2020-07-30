import os
import sys
import shutil
import tarfile
import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_extract.util import load_metadata
from ..integration_tests.test_cli import write_fake_movie
from moseq2_extract.helpers.data import get_selected_sessions, load_h5s, check_completion_status, \
                build_manifest, copy_manifest_results, handle_extract_metadata, get_pca_uuids, build_index_dict

class TestHelperData(TestCase):

    def test_check_completion_status(self):
        test_file = 'data/proc/results_00.yaml'
        assert check_completion_status(test_file) == True

        tmp_file = 'data/test_file.yaml' # non-existent
        assert check_completion_status(tmp_file) == False

    def test_get_pca_uuids(self):
        test_file = 'data/proc/results_00.yaml'

        with open(test_file, 'r') as f:
            dict = yaml.safe_load(f)

        uuids = get_pca_uuids([dict], '', [dict['uuid']])
        print(uuids)

        assert len(uuids) == 1
        assert uuids[0] == dict['uuid']

        uuids = get_pca_uuids([dict], '', None)
        assert len(uuids) == 1
        assert uuids[0] == dict['uuid']

    def test_build_index_dict(self):
        pca_file = ''
        filter = ('SessionName')

        test_file = 'data/proc/results_00.yaml'

        with open(test_file, 'r') as f:
            dict = yaml.safe_load(f)

        test_files = [('data/proc/results_00.h5',
                       'data/proc/results_00.yaml',
                       dict)]

        index = build_index_dict(test_files, pca_file, filter)

        assert len(index['files']) == 1
        assert index['pca_path'] == ''
        assert index['files'][0]['uuid'] == dict['uuid']

    def test_get_selected_sessions(self):

        to_extract = ['test1', 'test2', 'test3', 'test4', 'test5']

        stdin = 'data/stdin.txt'

        test_ext = get_selected_sessions(to_extract, True)

        assert test_ext == to_extract

        with open(stdin, 'w') as f:
            f.write('1-4, e2')
        f.close()

        sys.stdin = open(stdin)

        test_ext2 = get_selected_sessions(to_extract, False)

        assert test_ext2 == ['test1', 'test3', 'test4']
        os.remove(stdin)

    def test_load_h5s(self):

        test_fb_file = 'data/feedback_ts.txt'
        with open('data/depth_ts.txt', 'r') as f:
            with open(test_fb_file, 'w') as g:
                g.write(f.read())

        assert os.path.exists(test_fb_file)
        metadata = load_metadata('data/metadata.json')
        to_load = [(metadata, 'data/proc/results_00.h5')]
        loaded = load_h5s(to_load)

        assert len(loaded) > 0
        os.remove(test_fb_file)

    def test_build_manifest(self):
        metadata = load_metadata('data/metadata.json')
        to_load = [(metadata, 'data/proc/results_00.h5')]

        test_fb_file = 'data/feedback_ts.txt'
        with open('data/depth_ts.txt', 'r') as f:
            with open(test_fb_file, 'w') as g:
                g.write(f.read())

        assert os.path.exists(test_fb_file)
        loaded = load_h5s(to_load)

        manifest = build_manifest(loaded, 'results_00')
        assert isinstance(manifest, dict)
        os.remove(test_fb_file)

    def test_copy_manifest_results(self):
        metadata = load_metadata('data/metadata.json')
        to_load = [(metadata, 'data/proc/results_00.h5')]
        output_dir = 'data/tmp/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        test_fb_file = 'data/feedback_ts.txt'
        with open('data/depth_ts.txt', 'r') as f:
            with open(test_fb_file, 'w') as g:
                g.write(f.read())

        assert os.path.exists(test_fb_file)
        loaded = load_h5s(to_load)

        manifest = build_manifest(loaded, 'results_00')

        copy_manifest_results(manifest, output_dir)

        for p in os.listdir(output_dir):
            assert os.path.isfile(os.path.join(output_dir, p))

        shutil.rmtree(output_dir)

    def test_handle_extract_metadata(self):
        dirname = 'data/'
        tmp_file = 'data/test_vid.tar.gz'
        write_fake_movie(tmp_file)

        with tarfile.open(tmp_file, "w:gz") as tar:
            tar.add(dirname, arcname='test_vid.dat')
            tar.add(os.path.join(dirname, 'metadata.json'), arcname='metadata.json')
            tar.add(os.path.join(dirname, 'depth_ts.txt'), arcname='depth_ts.txt')

        input_file, acq_metadata, timestamps, alternate_correct, tar = handle_extract_metadata(tmp_file, dirname)

        assert isinstance(acq_metadata, dict)
        assert len(timestamps.shape) == 1
        assert tar != None
        assert alternate_correct == False

        os.remove(tmp_file)

        tmp_file = 'data/test_vid.dat'
        write_fake_movie(tmp_file)

        input_file, acq_metadata, timestamps, alternate_correct, tar = handle_extract_metadata(tmp_file, dirname)

        assert isinstance(acq_metadata, dict)
        assert len(timestamps.shape) == 1
        assert tar == None
        assert alternate_correct == False

        os.remove(tmp_file)