'''

Interactive ROI/Extraction Bokeh visualization functions.

'''

import warnings
from bokeh.models import Div
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from moseq2_extract.util import get_strels
from moseq2_extract.io.video import load_movie_data
from moseq2_extract.extract.extract import extract_chunk
from moseq2_extract.extract.proc import apply_roi, threshold_chunk

def show_extraction(input_file, video_file):
    '''

    Visualization helper function to display manually triggered extraction.
    Function will facilitate visualization through creating a HTML div to display
    in a jupyter notebook or web page.

    Parameters
    ----------
    input_file (str): session name to display.
    video_file (str): path to video to display

    Returns
    -------
    '''

    video_div = f'''
                    <h2>{input_file}</h2>
                    <video
                        src="{video_file}"; alt="{video_file}"; 
                        height="450"; width="450"; preload="auto";
                        style="float: left; type: "video/mp4"; margin: 0px 10px 10px 0px;
                        border="2"; autoplay controls loop>
                    </video>
                '''

    div = Div(text=video_div, style={'width': '100%'})
    show(div)

def bokeh_plot_helper(bk_fig, image):
    '''

    Helper function that creates the Bokeh image gylphs in the
    created canvases/figures.

    Parameters
    ----------
    bk_fig (Bokeh figure): figure canvas to draw image/glyph on
    image (2D np.array): image to draw.

    Returns
    -------
    '''

    bk_fig.x_range.range_padding = bk_fig.y_range.range_padding = 0
    if isinstance(image, dict):
        bk_fig.image(source=image, image='image', x='x', y='y', dw='dw', dh='dh', palette="Spectral11")
    else:
        bk_fig.image(image=[image],
                     x=0,
                     y=0,
                     dw=image.shape[1],
                     dh=image.shape[0],
                     palette="Spectral11")

def plot_roi_results(input_file, config_data, session_key, session_parameters, bground_im, roi, minmax_heights, fn):
    '''
    Main ROI plotting function that uses Bokeh to facilitate 3 interactive plots.
    Plots the background image, and an axis-connected plot of the ROI,
    and an independent plot of the thresholded background subracted segmented image.

    Parameters
    ----------
    input_file (str): path to current session
    config_data (dict): Extraction configuration parameters
    session_key (str): current session name (key in session_parameters)
    session_parameters (dict): current session parameters
    bground_im (2D np.array): Computed session background
    roi (2D np.array): Computed ROI based on given depth ranges
    minmax_heights (tuple or ipywidget IntRangeSlider): min and max mouse heights in segmented, background subtracted frame
    fn (int or ipywidget IntSlider): Current frame number to display

    Returns
    -------
    '''
    # ignore flip classifier sklearn version warnings
    warnings.filterwarnings('ignore')

    # set bokeh tools
    tools = 'pan, box_zoom, wheel_zoom, hover, reset'

    # update adjusted min and max heights
    config_data['min_height'] = minmax_heights[0]
    config_data['max_height'] = minmax_heights[1]

    # update current session parameters
    session_parameters[session_key] = config_data

    # get segmented frame
    raw_frame = load_movie_data(input_file, fn)
    curr_frame = (bground_im - raw_frame)

    # filter out regions outside of ROI
    filtered_frames = apply_roi(curr_frame, roi)[0]
    filtered_frames = threshold_chunk(filtered_frames, minmax_heights[0], minmax_heights[1]).astype('uint8')

    # Get overlayed ROI
    overlay = bground_im.copy()
    overlay[roi != True] = 0

    # Plot Background
    bg_fig = figure(title="Background",
                    tools=tools,
                    tooltips=[("(x,y)", "($x{0.1f}, $y{0.1f})"), ("value", "@image"), ('roi', '@roi')],
                    output_backend="webgl")

    data = dict(image=[bground_im],
                roi=[roi],
                x=[0],
                y=[0],
                dw=[bground_im.shape[1]],
                dh=[bground_im.shape[0]])

    bokeh_plot_helper(bg_fig, data)

    # plot overlayed roi
    overlay_fig = figure(title="Overlayed ROI",
                         x_range=bg_fig.x_range,
                         y_range=bg_fig.y_range,
                         tools=tools,
                         tooltips=[("(x,y)", "($x{0.1f}, $y{0.1f})"), ("value", "@image")],
                         output_backend="webgl")

    bokeh_plot_helper(overlay_fig, overlay)

    # plot segmented frame
    segmented_fig = figure(title=f"Segmented Frame #{fn}",
                           tools=tools,
                           tooltips=[("(x,y)", "($x{0.1f}, $y{0.1f})"), ("value", "@image")],
                           output_backend="webgl")

    bokeh_plot_helper(segmented_fig, filtered_frames)

    # plot crop rotated frame
    cropped_fig = figure(title=f"Crop-Rotated Frame #{fn}",
                         tools=tools,
                         tooltips=[("(x,y)", "($x{0.1f}, $y{0.1f})"), ("value", "@image")],
                         output_backend="webgl")

    # prepare extraction metadatas
    str_els = get_strels(config_data)
    config_data['tracking_init_mean'] = None
    config_data['tracking_init_cov'] = None

    # extract crop-rotated selected frame
    result = extract_chunk(**config_data,
                           **str_els,
                           chunk=raw_frame.copy(),
                           roi=roi,
                           bground=bground_im,
                           )

    bokeh_plot_helper(cropped_fig, result['depth_frames'][0])

    # Create 2x2 grid plot
    gp = gridplot([[bg_fig, overlay_fig],
                   [segmented_fig, cropped_fig]],
                    plot_width=300, plot_height=300)
    show(gp)
