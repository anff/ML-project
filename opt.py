import plotly.io as pio
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning) #, module="xgboost")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)


def set_env():
    if not os.path.exists(GlobalConst.dir_img):
        os.mkdir(GlobalConst.dir_img)
    #
    # layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')
    # fig = go.Figure(layout=layout)
    # templated_fig = pio.to_templated(fig)
    # pio.templates['my_template'] = templated_fig.layout.template
    # pio.templates.default = 'my_template'


class GlobalConst:
    model = 'lstm'
    slide_window = 20
    dir_img = 'image'
