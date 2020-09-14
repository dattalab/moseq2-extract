'''

Ipywidgets used to facilitate interactive ROI detection

'''

import ipywidgets as widgets
from ipywidgets import HBox, VBox

class InteractiveROIWidgets:
    '''
    Class that contains Ipywidget widgets and layouts to facilitate interactive ROI finding functionality.
    This class is extended by the controller class InteractiveFindRoi. 
    '''
    
    def __init__(self):
        '''
        Initializing all the ipywidgets widgets in a new context.
        '''

        style = {'description_width': 'initial'}

        self.label_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='100%')
        self.layout_hidden = widgets.Layout(visibility='hidden', display='none')
        self.layout_visible = widgets.Layout(visibility='visible', display='inline-flex')

        # session select widget
        self.sess_select = widgets.Dropdown(options=[], description='Session:', disabled=False)

        # roi widgets
        self.roi_label = widgets.Label(value="ROI Parameters", layout=self.label_layout)
        self.bg_roi_depth_range = widgets.IntRangeSlider(value=[650, 750], min=0, max=1500, step=5,
                                                    description='Included BG Depth Range', continuous_update=False, style=style)
        self.dilate_iters = widgets.IntSlider(value=0, min=0, max=25, step=1, description='Dilate Iters:',
                                        continuous_update=False, style=style)
        self.frame_num = widgets.IntSlider(value=0, min=0, max=1000, step=1, description='Current Frame:',
                                    disabled=False, continuous_update=False, style=style)

        self.toggle_autodetect = widgets.ToggleButton(value=False, description='Autodetect Depth Range', disabled=False,
                                                button_style='info', tooltip='Auto-detect depths', layout=self.label_layout)

        # extract widgets
        self.ext_label = widgets.Label(value="Extract Parameters", layout=self.label_layout)
        self.minmax_heights = widgets.IntRangeSlider(value=[10, 100], min=0, max=300, step=1,
                                                description='Mouse Heights to Capture', style=style, continuous_update=False)
        self.frame_range = widgets.IntRangeSlider(value=[0, 300], min=0, max=3000, step=30,
                                            description='Number of Frames to Extract', style=style, continuous_update=False)

        # check all button label
        self.checked_lbl = widgets.Label(value="Passing Sessions", layout=self.label_layout, button_style='info')

        # buttons
        self.save_parameters = widgets.Button(description='Save Parameters', disabled=False, tooltip='Save Parameters')
        self.check_all = widgets.Button(description='Check All Sessions', disabled=False,
                                tooltip='Extract full session using current parameters')

        self.checked_list = widgets.SelectMultiple(options=list(), value=[], description='', disabled=True)

        # groupings
        # ui widgets
        self.roi_tools = VBox([self.roi_label, self.bg_roi_depth_range, self.dilate_iters, self.frame_num])
        self.extract_tools = VBox([self.ext_label, self.minmax_heights, self.frame_range, self.toggle_autodetect])

        self.box_layout = widgets.Layout(display='inline-flex', flex_flow='row nowrap', justify_content='space-around',
                                    align_items='center', width='100%')

        self.button_box = VBox([HBox([self.check_all, self.save_parameters]), self.checked_lbl, self.checked_list])

        self.ui_tools = HBox([self.roi_tools, self.extract_tools, self.button_box], layout=self.box_layout)