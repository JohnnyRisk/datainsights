import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


def transform_data(val_data,ticker_list,encoder):    
    master_list=[['factor1', 'factor2', 'factor3', 'factor4', 'factor5', 'factor6', 'factor7']]
    for time_step in range(val_data[0].shape[0]):
        stock =[]
        for stock_num, ticker in enumerate(ticker_list):
            stock.append(encoder.predict(val_data[stock_num][time_step:(time_step+1),:,:])[0].tolist())
        master_list.append(('Timestep'+str(time_step), stock))
    return master_list

def transform_data2(val_data,ticker_list,encoder):
    dat_min=np.min(encoder.predict(np.array(val_data).reshape(620,20,5)),axis=0)
    dat_max=np.max(encoder.predict(np.array(val_data).reshape(620,20,5)),axis=0)
    
    master_list=[['factor1', 'factor2', 'factor3', 'factor4', 'factor5', 'factor6', 'factor7']]
    for time_step in range(val_data[0].shape[0]):
        stock =[]
        for stock_num, ticker in enumerate(ticker_list):
            stock.append(((encoder.predict(val_data[stock_num][time_step:(time_step+1),:,:])[0]-dat_min)/(dat_max - dat_min)).tolist())
        master_list.append(('Timestep'+str(time_step), stock))
    return master_list


def single_decoded_plot(encodings,decoder,title):
    decoded=decoder.predict(encodings.reshape((-1,1)).T)
    plt.figure(figsize=(20, 2))
    plt.suptitle(title, fontsize=14, fontweight='bold')
    j=5
    for n_feature in range(j):
        ax = plt.subplot(1, j, n_feature+1)
        plt.plot(np.array(range(decoded.shape[1])),
                 decoded[0,:,n_feature])
    plt.show()


def plot_time_ticker_pair(val_data,sequence_autoencoder,time_step1,ticker1,time_step2,ticker2):
    time_ticker1 =(time_step1, ticker1)
    time_ticker2 = (time_step2, ticker2)

    time_tick=[time_ticker1, time_ticker2]

    features = ['Open', 'High', 'Low','Close','Volume']

    ticker_to_num ={'SPY':0,'GOOG':1,'AAPL':2,'JCP':3,'XOM':4}

    for tt in time_tick:
        stock_name = tt[1]
        num = ticker_to_num[stock_name]
        plt.figure(figsize=(20, 2))
        n=tt[0]
        plt.suptitle(stock_name+' Embeddings', fontsize=14, fontweight='bold')
        j=len(features)
        for n_feature, feature in enumerate(features):
            ax = plt.subplot(1, j, n_feature+1)
            plt.plot(np.array(range(val_data[num][n,:,n_feature].shape[0])),
                     val_data[num][n,:,n_feature],
                     np.array(range(val_data[num][n,:,n_feature].shape[0])),
                     sequence_autoencoder.predict(val_data[num])[n,:,n_feature])
        plt.show()

def plot_spider_chart(data):
    N = 7
    theta = radar_factory(N, frame='polygon')
    spoke_labels = data.pop(0)

    fig, axes = plt.subplots(figsize=(9, 9), nrows=62, ncols=2,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    fig.set_figheight(300)

    colors = ['b', 'r', 'g', 'm', 'y']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axes.flatten(), data):
        ax.set_rgrids([0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    ax = axes[0, 0]
    labels = ('SPY', 'GOOG', 'AAPL', 'JCP', 'XOM')
    legend = ax.legend(labels, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, '7-Factors of 5 Stocks',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()