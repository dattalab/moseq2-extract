'''

Interactive ROI detection functionality. This module utilizes the widgets from
the widgets.py file to facilitate the real-time interaction.

'''

import os
import cv2
import math
import numpy as np
from math import isclose
import ruamel.yaml as yaml
import ipywidgets as widgets
from ipywidgets import interact, fixed
from os.path import dirname, basename, join
from IPython.display import display, clear_output
from moseq2_extract.helpers.extract import process_extract_batches
from moseq2_extract.extract.proc import get_roi, get_bground_im_file
from moseq2_extract.util import get_bucket_center, get_strels, select_strel
from moseq2_extract.interactive.view import plot_roi_results, show_extraction
from moseq2_extract.interactive.widgets import sess_select, save_parameters, bg_roi_depth_range, minmax_heights, \
                                        check_all, checked_list, checked_lbl, frame_num, frame_range, \
                                        dilate_iters, ui_tools, layout_visible

def test_all_sessions(session_dict, config_data, session_parameters):
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

    all_results = {}
    # test saved config data parameters on all sessions
    for sessionName, sessionPath in session_dict.items():
        if sessionName not in checked_list.value:
            # Get background image for each session and test the current parameters on it
            bground_im = get_bground_im_file(sessionPath)
            sess_res = get_roi_and_depths(bground_im, config_data, False)

            if not sess_res['flagged']:
                session_parameters[sessionName] = config_data

            all_results[sessionName] = sess_res['flagged']

    return all_results

def interactive_find_roi_session_selector(session, config_data, session_parameters):
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
    out = widgets.interactive_output(interactive_depth_finder, {'session': fixed(session),
                                                          'bground_im': fixed(bground_im),
                                                          'config_data': fixed(config_data),
                                                          'session_parameters': fixed(session_parameters),
                                                          'dr': bg_roi_depth_range,
                                                          'di': dilate_iters})
    display(ui_tools, out)

    def check_all_sessions(b):
        '''
        Callback function to run the ROI area comparison test on all the existing sessions.
        Saving their individual session parameter sets in the session_parameters dict in the process.

        Parameters
        ----------
        b (button event): User click

        Returns
        -------
        '''

        check_all.description = 'Checking...'

        res = test_all_sessions(sess_select.options, config_data, session_parameters)

        if all(list(res.values())) == False:
            check_all.button_style = 'success'
            check_all.icon = 'check'
        else:
            check_all.button_style = 'danger'
        check_all.description = 'Check All Sessions'

        save_parameters.layout = layout_visible

    check_all.on_click(check_all_sessions)

    def save_clicked(b):
        '''
        Callback function to save the current session_parameters dict into
        the file of their choice (given in the top-most wrapper function).

        Parameters
        ----------
        b (button event): User click

        Returns
        -------
        '''

        with open(config_data['session_config_path'], 'w+') as f:
            yaml.safe_dump(session_parameters, f)

        with open(config_data['config_file'], 'w+') as f:
            yaml.safe_dump(config_data, f)

        save_parameters.button_style = 'success'
        save_parameters.icon = 'check'

    save_parameters.on_click(save_clicked)

def interactive_depth_finder(session, bground_im, config_data, session_parameters, dr, di):
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

    save_parameters.button_style = 'primary'
    save_parameters.icon = 'none'

    # autodetect reference depth range and min-max height values at launch
    if config_data['inital']:
        config_data, results = get_roi_and_depths(bground_im, config_data)
        config_data['inital'] = False

        # set initial frame range tuple value
        config_data['frame_range'] = frame_range.value

        # update sliders with autodetected values
        bg_roi_depth_range.value = config_data['bg_roi_depth_range']
        minmax_heights.value = [config_data['min_height'], config_data['max_height']]
    else:
        # test updated parameters
        config_data['bg_roi_depth_range'] = (int(dr[0]), int(dr[1]))
        config_data['dilate_iterations'] = di
        config_data, results = get_roi_and_depths(bground_im, config_data, config_data['inital'])

    # clear output to update view
    clear_output()

    # display validation indicator
    label_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='100%')
    indicator = widgets.Label(value="", font_size=50, layout=label_layout)

    keys = list(sess_select.options.keys())

    # set indicator
    if results['flagged']:
        indicator.value = r'\(\color{red} {Flagged}\)'
    else:
        indicator.value = r'\(\color{green} {Passing}\)'
        checked_list.value = list(set(list(checked_list.value) + [keys[sess_select.index]]))
        session_parameters[keys[sess_select.index]] = config_data
        checked_lbl.value = f'Passing Sessions: {len(list(checked_list.value))}/{len(checked_list.options)}'

    display(indicator)

    # display graphs
    out = widgets.interactive_output(plot_roi_results, {'input_file': fixed(session),
                                                        'config_data': fixed(config_data),
                                                        'session_parameters': fixed(session_parameters),
                                                        'session_key': fixed(keys[sess_select.index]),
                                                        'bground_im': fixed(bground_im),
                                                        'roi': fixed(results['roi']),
                                                        'minmax_heights': minmax_heights,
                                                        'fn': frame_num})
    display(out)

    def update_minmax_config(event):
        '''
        Callback function to update config dict with current UI min/max height range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        config_data['min_height'] = minmax_heights.value[0]
        config_data['max_height'] = minmax_heights.value[1]

    minmax_heights.observe(update_minmax_config, names='value')

    def update_config_fr(event):
        '''
        Callback function to update config dict with current UI depth range values

        Parameters
        ----------
        event (ipywidget callback): Any user interaction.

        Returns
        -------
        '''

        config_data['frame_range'] = frame_range.value

    frame_range.observe(update_config_fr, names='value')

    # manual extract API
    interact_ext = interact.options(manual=True, manual_name="Extract Sample")

    # Generates a button below the bokeh plots
    interact_ext(get_extraction,
                 input_file=fixed(session),
                 config_data=fixed(config_data),
                 bground_im=fixed(bground_im),
                 roi=fixed(results['roi']))


def get_roi_and_depths(bground_im, config_data, autodetect=True):
    '''
    Performs bucket centroid estimation to find the coordinates to use as the true depth value.
    The true depth will be used to estimate the background depth_range, then it will update the
    widget values in real time.

    Parameters
    ----------
    bground_im (2D np.array): Computed session background
    config_data (dict): Extraction configuration parameters
    autodetect (bool): boolean for whether to compute the true depth

    Returns
    -------
    results (dict): dict that contains computed information. E.g. its ROI, and if it was flagged.
    '''

    # initialize results dict
    results = {'flagged': False}

    if autodetect:
        # Get max depth as a thresholding limit (this would be the DTD if it already was computed)
        limit = np.max(bground_im)

        # Threshold image to find depth at bucket center: the true depth
        cX, cY = get_bucket_center(bground_im, limit, threshold=bground_im.mean())

        # True depth is at the center of the bucket
        true_depth = bground_im[cX][cY]

        # Center the depth ranges around the true depth
        bg_roi_range_min = true_depth - 100
        bg_roi_range_max = true_depth + 100

        config_data['bg_roi_depth_range'] = (bg_roi_range_min, bg_roi_range_max)

    strel_dilate = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_dilate']))
    strel_erode = select_strel(config_data['bg_roi_shape'], tuple(config_data['bg_roi_erode']))

    # Get ROI
    rois, plane, bboxes, _, _, _ = get_roi(bground_im,
                                           **config_data,
                                           strel_dilate=strel_dilate,
                                           strel_erode=strel_erode
                                           )

    if autodetect:
        # Get pixel dims from bounding box
        xmin = bboxes[0][0][1]
        xmax = bboxes[0][1][1]

        ymin = bboxes[0][0][0]
        ymax = bboxes[0][1][0]

        # bucket width in pixels
        pixel_width = xmax - xmin
        pixel_height = ymax - ymin

        pixels_per_inch = pixel_width / config_data['arena_width']
        config_data['pixels_per_inch'] = float(pixels_per_inch)
    else:
        pixels_per_inch = config_data['pixels_per_inch']

    # Corresponds to a rough pixel area estimate
    r = float(cv2.countNonZero(rois[0].astype('uint8')))
    config_data['pixel_area'] = r

    # Compute arena area
    if config_data['arena_shape'] == 'ellipse':
        area = math.pi * (config_data['arena_width'] / 2) ** 2
    elif 'rect' in config_data['arena_shape']:
        estimated_height = pixel_height / pixels_per_inch
        area = config_data['arena_width'] * estimated_height

    # Compute pixel per metric
    area_px_per_inch = r / area / pixels_per_inch

    try:
        assert isclose(config_data['pixel_area'], r, abs_tol=10e3)
    except AssertionError:
        if area_px_per_inch < pixels_per_inch:
            results['flagged'] = True

    # Save ROI
    results['roi'] = rois[0]

    return config_data, results

def get_extraction(input_file, config_data, bground_im, roi):
    '''

    Parameters
    ----------
    input_file
    config_data (dict): Extraction configuration parameters
    bground_im (2D np.array): Computed session background
    roi
    fr

    Returns
    -------

    '''

    str_els = get_strels(config_data)

    output_dir = dirname(input_file)
    outpath = 'test_extraction'
    view_path = join(output_dir, outpath+'.mp4')

    frame_batches = [range(config_data['frame_range'][0], config_data['frame_range'][1])]

    if os.path.exists(view_path):
        os.remove(view_path)

    # load chunk to display
    process_extract_batches(input_file, config_data, bground_im, roi, frame_batches,
                            frame_num.value, str_els, output_dir, outpath)

    # display extracted video as HTML Div using Bokeh
    show_extraction(basename(dirname(input_file)), view_path)