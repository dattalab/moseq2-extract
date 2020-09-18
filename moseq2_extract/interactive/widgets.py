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
        self.sess_select = widgets.Dropdown(options=[], description='Session:', disabled=False, continuous_update=True)

        # roi widgets
        self.roi_label = widgets.Label(value="ROI Parameters", layout=self.label_layout)
        self.bg_roi_depth_range = widgets.IntRangeSlider(value=[650, 750], min=0, max=1500, step=5,
                                                    description='Depth Range', continuous_update=False, style=style)
        self.dilate_iters = widgets.IntSlider(value=0, min=0, max=25, step=1, description='Dilate Iters:',
                                        continuous_update=False, style=style)
        self.frame_num = widgets.IntSlider(value=0, min=0, max=1000, step=1, description='Current Frame:',
                                    tooltip='Processed Frame Index to Display',
                                    disabled=False, continuous_update=False, style=style)

        self.toggle_autodetect = widgets.Checkbox(value=False, description='Autodetect Depth Range',
                                                  tooltip='Auto-detect depths', layout=widgets.Layout(display='none'))

        # extract widgets
        self.ext_label = widgets.Label(value="Extract Parameters", layout=self.label_layout)
        self.minmax_heights = widgets.IntRangeSlider(value=[13, 100], min=0, max=300, step=1,
                                                description='Min-Max Mouse Height', style=style, continuous_update=False)
        self.frame_range = widgets.IntRangeSlider(value=[0, 300], min=0, max=3000, step=30,
                                            tooltip='Frames to Extract Sample',
                                            description='Frame Range', style=style, continuous_update=False)

        # check all button label
        self.checked_lbl = widgets.Label(value="Passing Sessions", layout=self.label_layout, button_style='info')

        self.message = widgets.Label(value="", font_size=50, layout=self.label_layout)

        # buttons
        self.save_parameters = widgets.Button(description='Save Parameters', disabled=False, tooltip='Save Parameters')
        self.check_all = widgets.Button(description='Check All Sessions', disabled=False,
                                tooltip='Extract full session using current parameters')

        self.extract_button = widgets.Button(description='Extract Sample', disabled=False, layout=self.label_layout,
                                             tooltip='Preview extraction output')
        self.mark_passing = widgets.Button(description='Mark Passing', disabled=False, layout=self.label_layout,
                                           tooltip='Mark current session as passing')

        self.checked_list = widgets.SelectMultiple(options=list(), value=[], description='', disabled=True)

        # groupings
        # ui widgets
        self.roi_tools = VBox([self.roi_label, self.bg_roi_depth_range, self.dilate_iters, self.frame_num])
        self.extract_tools = VBox([self.ext_label, self.minmax_heights, self.frame_range, self.extract_button])

        self.box_layout = widgets.Layout(display='inline-flex', flex_flow='row nowrap', justify_content='space-around',
                                    align_items='center', width='100%')

        self.button_box = VBox([HBox([self.check_all, self.save_parameters]), self.checked_lbl, self.checked_list, self.mark_passing])

        self.ui_tools = VBox([HBox([self.roi_tools, self.extract_tools, self.button_box], layout=self.box_layout), self.message])