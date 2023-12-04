#! Users/Kathy/anaconda3/envs/seaflow/bin/python3
## script that contains graphing helper functions

# libraries to import 
import re
import plotly.graph_objects as go
import matplotlib.pyplot as plt

## easily aestheticize plotly figures, especially facet ones
def pretty_plot(fig, rescale_x=False, rescale_y=False, x_label="", y_label=""):
    # rescale each x axis if yes
    if rescale_x:
        # rescale each axis to be different 
        for k in fig.layout:
            if re.search('xaxis[1-9]+', k):
                fig.layout[k].update(matches=None)
        
        # add x axis back in
        fig.update_xaxes(showticklabels=True, 
                      tickangle=0, tickfont=dict(size=10))
    # rescale each y axis if yes
    if rescale_y:
        # rescale each axis to be different 
        for k in fig.layout:
            if re.search('yaxis[1-9]+', k):
                fig.layout[k].update(matches=None)
        
        # add y axis back in
        fig.update_yaxes(showticklabels=True, col=2)

    # shorten default subplot titles
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

#     # fix annotations to make them horizontal
#     for annotation in fig['layout']['annotations']: 
#         annotation['textangle']= 0

    # horizontal colorbar
    #fig.update_layout(coloraxis_colorbar_x=-0.15)

    # #decrease font size 
    fig.update_annotations(font_size=10)

    # hide subplot y-axis titles and x-axis titles
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.text = ''
    # keep all other annotations and add single y-axis and x-axis title:
    fig.update_layout(
        # keep the original annotations and add a list of new annotations:
        annotations = list(fig.layout.annotations) + 
        [go.layout.Annotation(
                x=-0.07,
                y=0.5,
                font=dict(
                    size=16, color = 'black'
                ),
                showarrow=False,
                text=y_label,
                textangle=-90,
                xref="paper",
                yref="paper"
            )
        ] +
        [go.layout.Annotation(
                x=0.5,
                y=-0.1,
                font=dict(
                    size=16, color = 'black'
                ),
                showarrow=False,
                text=x_label,
                textangle=-0,
                xref="paper",
                yref="paper"
            )
        ]
    )
    return(fig)

## helper function for double y axes in matplotlib (still need to adjust to not be hardcoded)
def plt_double_axis(x, y1, y2, x_title=None, y1_title=None, y2_title=None):
    fig, axs = plt.subplots(figsize=(20,8))

    ax2 = axs.twinx()
    axs.plot(x, y1, 'g-')
    ax2.plot(x, y2, 'b-')

    axs.set_xlabel(x_title)
    axs.set_ylabel(y1_title, color='g')
    ax2.set_ylabel(y2_title, color='b')

    # returns output of figure
    return(fig)

from diel_tools import find_night
from tsd_functions import interp_by_time
import matplotlib.dates as mdates
# helper function to show raw data of a cruise with day and night plotted 
## cruise=pd.dataframe with pop columns (pro/syn only), col=pd.series column in dataframe for data to plot, name=string for cruise name
def show_raw_diel(cruise, col, name):
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10,8), sharex=True)
    x=cruise[cruise['pop']=='prochloro']['time']
    y=cruise[cruise['pop']=='prochloro'][col]
    x1=cruise[cruise['pop']=='synecho']['time']
    y1=cruise[cruise['pop']=='synecho'][col]
    l1 = axs[0].plot(x,y, label='pro', marker='.')
    l2 = axs[1].plot(x1,y1,c='green', label='syn', marker='.')
    lns = l1+l2
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns, labs, loc=0)
    # adjust layout
    plt.tight_layout()
    # rotate xticks
    axs[0].set_ylabel(f'Pro {col}')
    axs[1].set_ylabel(f'Syn {col}')
    axs[0].set_title(f'{col} for {name}')
    # fill in night
    ax0 = axs[0].twinx()
    ax1 = axs[1].twinx()
    # get full days from interpolated data
    # plot interpolated data results
    cruise_int_pro = find_night(interp_by_time(cruise, 'prochloro').reset_index())
    cruise_int_syn = find_night(interp_by_time(cruise, 'synecho').reset_index())
    ax0.fill_between(cruise_int_pro['time'], 0, 1, where=cruise_int_pro['night'] != 'day',
                        color='gray', alpha=0.3, transform=ax0.get_xaxis_transform())
    ax1.fill_between(cruise_int_syn['time'], 0, 1, where=cruise_int_syn['night'] != 'day',
                        color='gray', alpha=0.3, transform=ax1.get_xaxis_transform())

    axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gcf().autofmt_xdate()
    # hide every other label
    for label in axs[1].xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    # update font size
    plt.rcParams.update({'font.size':15})