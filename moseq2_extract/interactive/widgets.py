'''

Ipywidgets used to facilitate interactive ROI detection

'''

import ipywidgets as widgets
from ipywidgets import HBox, VBox

style = {'description_width': 'initial'}

label_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='100%')

# session select widget
sess_select = widgets.Dropdown(options=[], description='Session:', disabled=False)

# roi widgets
roi_label = widgets.Label(value="ROI Parameters", layout=label_layout)
bg_roi_depth_range = widgets.IntRangeSlider(value=[650, 750], min=0, max=1500, step=5,
                                            description='Included Depth Range', continuous_update=False, style=style)
dilate_iters = widgets.IntSlider(value=0, min=0, max=25, step=1, description='Dilate Iters:',
                                 continuous_update=False, style=style)
frame_num = widgets.IntSlider(value=0, min=0, max=1000, step=1, description='Current Frame:',
                              disabled=False, continuous_update=False, style=style)

# extract widgets
ext_label = widgets.Label(value="Extract Parameters", layout=label_layout)
minmax_heights = widgets.IntRangeSlider(value=[10, 100], min=0, max=300, step=1,
                                        description='Depth Values to Capture', style=style, continuous_update=False)
frame_range = widgets.IntRangeSlider(value=[0, 300], min=0, max=3000, step=30,
                                     description='Number of Frames to Extract', style=style, continuous_update=False)

# check all button label
checked_lbl = widgets.Label(value="Passing Sessions", layout=label_layout, button_style='info')

# buttons
save_parameters = widgets.Button(description='Save Parameters', disabled=False, tooltip='Save Parameters')
check_all = widgets.Button(description='Check All Sessions', disabled=False,
                           tooltip='Extract full session using current parameters')

checked_list = widgets.SelectMultiple(options=list(), value=[], description='', disabled=True)

# groupings
# ui widgets
roi_tools = VBox([roi_label, bg_roi_depth_range, dilate_iters, frame_num])
extract_tools = VBox([ext_label, minmax_heights, frame_range])

layout_hidden = widgets.Layout(visibility='hidden', display='none')
layout_visible = widgets.Layout(visibility='visible', display='inline-flex')
box_layout = widgets.Layout(display='inline-flex', flex_flow='row nowrap', justify_content='space-around',
                            align_items='center', width='100%')

button_box = VBox([HBox([check_all, save_parameters]), checked_lbl, checked_list])

ui_tools = HBox([roi_tools, extract_tools, button_box], layout=box_layout)