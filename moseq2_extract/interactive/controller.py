'''

Interactive ROI detection functionality. This module utilizes the widgets from
the widgets.py file to facilitate the real-time interaction.

'''

import os
import cv2
import math
import warnings
import numpy as np
from math import isclose
import ruamel.yaml as yaml
import ipywidgets as widgets
from ipywidgets import interact, fixed
from os.path import dirname, basename, join
from IPython.display import display, clear_output
from moseq2_extract.helpers.data import get_session_paths
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.interactive.widgets import InteractiveROIWidgets
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.interactive.view import plot_roi_results, show_extraction
from moseq2_extract.util import get_bucket_center, get_strels, select_strel, set_bground_to_plane_fit


class InteractiveFindRoi(InteractiveROIWidgets):

    def __init__(self, data_path, config_file, session_config):
        super().__init__()

        # Read default config parameters
        with open(config_file, 'r') as f:
            self.config_data = yaml.safe_load(f)
        
        self.session_config = session_config

        # Read individual session config if it exists
        if session_config is not None:
            if os.path.exists(session_config):
                with open(session_config, 'r') as f:
                    self.session_parameters = yaml.safe_load(f)
            else:
                warnings.warn('Session configuration file was not found. Generating a new one.')

                # Generate session config file if it does not exist
                session_config = join(dirname(config_file), 'session_config.yaml')
                self.session_parameters = {}
                with open(session_config, 'w+') as f:
                    yaml.safe_dump(self.session_parameters, f)
        
        self.all_results = {}

        self.config_data['session_config_path'] = session_config
        self.config_data['config_file'] = config_file

        # Update DropDown menu items
        self.sess_select.options = get_session_paths(data_path)
        self.checked_list.options = list(self.sess_select.options.keys())

        # Set toggle button callback
        self.toggle_autodetect.observe(self.toggle_button_clicked, names='value')

        # Set save parameters button callback
        self.save_parameters.on_click(self.save_clicked)

        # Set check all sessions button callback
        self.check_all.on_click(self.check_all_sessions)

        # Set min-max range slider callback
        self.minmax_heights.observe(self.update_minmax_config, names='value')

        # Set extract frame range slider
        self.frame_range.observe(self.update_config_fr, names='value')

    def toggle_button_clicked(self, b):
        '''
        Updates the true depth autodetection parameter
         such that the true depth is autodetected for each found session

        Parameters
        ----------
        b (ipywidgets Button): Button click event.

        Returns
        -------
        '''

        self.config_data['autodetect'] = self.toggle_autodetect.value

    def check_all_sessions(self, b):
        '''
        Callback function to run the ROI area comparison test on all the existing sessions.
        Saving their individual session parameter sets in the session_parameters dict in the process.

        Parameters
        ----------
        b (button event): User click

        Returns
        -------
        '''

        self.check_all.description = 'Checking...'

        self.test_all_sessions(self.sess_select.options)

        if all(list(self.all_results.values())) == False:
            self.check_all.button_style = 'success'
            self.check_all.icon = 'check'
        else:
            self.check_all.button_style = 'danger'
        self.check_all.description = 'Check All Sessions'

        self.save_parameters.layout = self.layout_visible

    def save_clicked(self, b):
        '''
        Callback function to save the current session_parameters dict into
        the file of their choice (given in the top-most wrapper function).

        Parameters
        ----------
        b (button event): User click

        Returns
        -------
        '''

        with open(self.config_data['session_config_path'], 'w+') as f:
            yaml.safe_dump(self.session_parameters, f)

        with open(self.config_data['config_file'], 'w+') as f:
            yaml.safe_dump(self.config_data, f)

        self.save_parameters.button_style = 'success'
        self.save_parameters.icon = 'check'

        tmp_options = self.sess_select.options.copy()
        for k, v in self.all_results.items():
            if not v:
                if len(tmp_options.keys()) > 1 and k in tmp_options.keys():
                    del tmp_options[k]
        
        self.sess_select.options = tmp_options
        self.checked_list.options = list(self.sess_select.options.keys())
        self.checked_list.value = list(self.sess_select.options.keys())[0]

    def update_minmax_config(self, event):
        '''
        Callback function to update config dict with current UI min/max height range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        self.config_data['min_height'] = self.minmax_heights.value[0]
        self.config_data['max_height'] = self.minmax_heights.value[1]

    def update_config_fr(self, event):
        '''
        Callback function to update config dict with current UI depth range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        self.config_data['frame_range'] = self.frame_range.value
    
    def test_all_sessions(self, session_dict):
        '''
        Helper function to test the current configurable UI values on all the
        sessions that were found.

        Parameters
        ----------
        session_dict (dict): dict of session directory names paired with their absolute paths.
        config_data (dict): ROI/Extraction configuration parameters.

        Returns
        -------
        all_results (dict): dict of session names and values used to indicate if a session was flagged,
        with their computed ROI for convenience.
        '''
        # test saved config data parameters on all sessions
        for sessionName, sessionPath in session_dict.items():
            if sessionName not in self.checked_list.value:
                # Get background image for each session and test the current parameters on it
                bground_im = get_bground_im_file(sessionPath)
                sess_res = self.get_roi_and_depths(bground_im, sessionPath)

                if not sess_res['flagged']:
                    self.session_parameters[sessionName] = self.config_data
                    self.checked_list.value = list(set(list(self.checked_list.value))) + [sessionName]
                    self.checked_lbl.value = f'Passing Sessions: {len(list(self.checked_list.value))}/{len(self.checked_list.options)}'
                else:
                    self.checked_list.value = list(self.checked_list.value).remove(sessionName)
                    self.checked_lbl.value = f'Passing Sessions: {len(list(self.checked_list.value))}/{len(self.checked_list.options)}'

                self.all_results[sessionName] = sess_res['flagged']
        
        if len(self.checked_list.value) == len(self.checked_list.options):
            self.message.value = 'All sessions passed with the current parameter set. \
            Save the parameters and move to the "Extract All" cell.'
        else:
            self.message.value = 'Some sessions were flagged. Save the parameter set for the current passing sessions, \
             then find and save the correct set for the remaining sessions.'
        

    def interactive_find_roi_session_selector(self, session):
        '''
        First function that is called to find the current selected session's background
        and display the widget interface.

        Parameters
        ----------
        session (str or ipywidget DropDownMenu): path to chosen session.
        config_data (dict): ROI/Extraction configuration parameters

        Returns
        -------
        '''

        bground_im = get_bground_im_file(session)
        clear_output()
        out = widgets.interactive_output(self.interactive_depth_finder, {'session': fixed(session),
                                                            'bground_im': fixed(bground_im),
                                                            'dr': self.bg_roi_depth_range,
                                                            'di': self.dilate_iters})
        display(self.sess_select, self.ui_tools, out)

    def interactive_depth_finder(self, session, bground_im, dr, di):
        '''
        Interactive helper function that updates that views whenever the depth range or
        dilation iterations sliders are changed.
        At initial launch, it will auto-detect the depth estimation, then it will preserve the parameters
        across session changes.

        Parameters
        ----------
        session (str or ipywidget DropDownMenu): path to input file
        bground_im (2D np.array): Computed session background
        config_data (dict): Extraction configuration parameters
        dr (tuple or ipywidget IntRangeSlider): Depth range to capture
        di (int or ipywidget IntSlider): Dilation iterations

        Returns
        -------
        '''

        if '.tar' in session:
            self.config_data['tar'] = True
        else:
            self.config_data['tar'] = False

        self.save_parameters.button_style = 'primary'
        self.save_parameters.icon = 'none'

        # Session selection dict key names
        keys = list(self.sess_select.options.keys())

        # Autodetect reference depth range and min-max height values at launch
        if self.config_data['autodetect']:
            results = self.get_roi_and_depths(bground_im, session)
            if not results['flagged']:
                self.config_data['autodetect'] = False

            # Update the session flag result
            self.all_results[keys[self.sess_select.index]] = results['flagged']

            # Set initial frame range tuple value
            self.config_data['frame_range'] = self.frame_range.value

            # Update sliders with autodetected values
            self.bg_roi_depth_range.value = self.config_data['bg_roi_depth_range']
            self.minmax_heights.value = [self.config_data['min_height'], self.config_data['max_height']]
        else:
            # Test updated parameters
            self.config_data['bg_roi_depth_range'] = (int(dr[0]), int(dr[1]))
            self.config_data['dilate_iterations'] = di
            
            # Update the session flag result
            results = self.get_roi_and_depths(bground_im, session)
            self.all_results[keys[self.sess_select.index]] = results['flagged']

        # Clear output to update view
        clear_output()

        # Display validation indicator
        indicator = widgets.Label(value="", font_size=50, layout=self.label_layout)

        # set indicator
        if results['flagged']:
            indicator.value = r'\(\color{red} {Flagged}\)'
            self.checked_list.value = [v for v in self.checked_list.value if v != keys[self.sess_select.index]]
            self.checked_lbl.value = f'Passing Sessions: {len(list(self.checked_list.value))}/{len(self.checked_list.options)}'
        else:
            indicator.value = r'\(\color{green} {Passing}\)'
            self.checked_list.value = list(set(list(self.checked_list.value) + [keys[self.sess_select.index]]))
            self.session_parameters[keys[self.sess_select.index]] = self.config_data
            self.checked_lbl.value = f'Passing Sessions: {len(list(self.checked_list.value))}/{len(self.checked_list.options)}'

        # Display extraction validation indicator
        display(indicator)

        out = widgets.interactive_output(plot_roi_results, {'input_file': fixed(session),
                                                            'config_data': fixed(self.config_data),
                                                            'session_parameters': fixed(self.session_parameters),
                                                            'session_key': fixed(keys[self.sess_select.index]),
                                                            'bground_im': fixed(bground_im),
                                                            'roi': fixed(results['roi']),
                                                            'minmax_heights': self.minmax_heights,
                                                            'fn': self.frame_num})
        # display graphs
        display(out)
        
        # manual extract API
        interact_ext = interact.options(manual=True, manual_name="Extract Sample")

        # Generates a button below the bokeh plots
        interact_ext(self.get_extraction,
                    input_file=fixed(session),
                    bground_im=fixed(bground_im),
                    roi=fixed(results['roi']))


    def get_roi_and_depths(self, bground_im, session):
        '''
        Performs bucket centroid estimation to find the coordinates to use as the true depth value.
        The true depth will be used to estimate the background depth_range, then it will update the
        widget values in real time.

        Parameters
        ----------
        bground_im (2D np.array): Computed session background
        session (str): path to currently processed session
        config_data (dict): Extraction configuration parameters

        Returns
        -------
        results (dict): dict that contains computed information. E.g. its ROI, and if it was flagged.
        '''

        # initialize results dict
        results = {'flagged': False}

        if self.config_data['autodetect']:
            # Get max depth as a thresholding limit (this would be the DTD if it already was computed)
            limit = np.max(bground_im)

            # Threshold image to find depth at bucket center: the true depth
            cX, cY = get_bucket_center(bground_im, limit, threshold=bground_im.mean())

            # True depth is at the center of the bucket
            true_depth = bground_im[cX][cY]

            # Get true depth range difference
            range_diff = 10**(len(str(int(true_depth)))-1)

            # Center the depth ranges around the true depth
            bg_roi_range_min = true_depth - range_diff
            bg_roi_range_max = true_depth + range_diff

            self.config_data['bg_roi_depth_range'] = (bg_roi_range_min, bg_roi_range_max)

        # Get relevant structuring elements
        strel_dilate = select_strel(self.config_data['bg_roi_shape'], tuple(self.config_data['bg_roi_dilate']))
        strel_erode = select_strel(self.config_data['bg_roi_shape'], tuple(self.config_data['bg_roi_erode']))

        # Get ROI
        rois, plane, bboxes, _, _, _ = get_roi(bground_im,
                                            **self.config_data,
                                            strel_dilate=strel_dilate,
                                            strel_erode=strel_erode
                                            )

        if self.config_data['use_plane_bground']:
            print('Using plane fit for background...')
            bground_im = set_bground_to_plane_fit(bground_im, plane, join(dirname(session, 'proc')))

        if self.config_data['autodetect']:
            # Get pixel dims from bounding box
            xmin = bboxes[0][0][1]
            xmax = bboxes[0][1][1]

            ymin = bboxes[0][0][0]
            ymax = bboxes[0][1][0]

            # bucket width in pixels
            pixel_width = xmax - xmin
            pixel_height = ymax - ymin

            if self.config_data.get('arena_width') is not None:
                pixels_per_inch = pixel_width / self.config_data['arena_width']
            elif self.config_data.get('true_height') is not None:
                pixels_per_inch = pixel_width / self.config_data['true_height']
            else:
                print('Error: you must enter your arena width or true camera height.')

            self.config_data['pixels_per_inch'] = float(pixels_per_inch)

            # Corresponds to a rough pixel area estimate
            r = float(cv2.countNonZero(rois[0].astype('uint8')))
            self.config_data['pixel_area'] = r
        else:
            pixels_per_inch = self.config_data['pixels_per_inch']
            # Corresponds to a rough pixel area estimate
            r = float(cv2.countNonZero(rois[0].astype('uint8')))

        if self.config_data.get('arena_width') is not None:
            # Compute arena area
            if self.config_data['arena_shape'] == 'ellipse':
                area = math.pi * (self.config_data['arena_width'] / 2) ** 2
            elif 'rect' in self.config_data['arena_shape']:
                estimated_height = pixel_height / pixels_per_inch
                area = self.config_data['arena_width'] * estimated_height

            # Compute pixel per metric
            self.config_data['area_px_per_inch'] = r / area / pixels_per_inch

        try:
            assert isclose(self.config_data['pixel_area'], r, abs_tol=50e2)
        except AssertionError:
            if self.config_data.get('area_px_per_inch', 0) < pixels_per_inch:
                results['flagged'] = True

        # Save ROI
        results['roi'] = rois[0]
        results['counted_pixels'] = r

        return results

    def get_extraction(self, input_file, bground_im, roi):
        '''
        Extracts selected frame range (with the currently set session parameters)
        and displays the extraction as a Bokeh HTML-embedded div.

        Parameters
        ----------
        input_file (str): Path to session to extract
        config_data (dict): Extraction configuration parameters.
        bground_im (2D np.array): Computed session background.
        roi (2D np.array): Computed Region of interest array to mask bground_im with.

        Returns
        -------
        '''

        # Get structuring elements
        str_els = get_strels(self.config_data)

        # Get output path for extraction video
        output_dir = dirname(input_file)
        outpath = 'extraction_preview'
        view_path = join(output_dir, outpath+'.mp4')

        # Get frames to extract
        frame_batches = [range(self.config_data['frame_range'][0], self.config_data['frame_range'][1])]

        # Remove previous preview
        if os.path.exists(view_path):
            os.remove(view_path)

        # load chunk to display
        process_extract_batches(input_file, self.config_data, bground_im, roi, frame_batches,
                                0, str_els, output_dir, outpath)

        # display extracted video as HTML Div using Bokeh
        show_extraction(basename(dirname(input_file)), view_path)