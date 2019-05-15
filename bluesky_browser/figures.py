import collections
import logging
from functools import partial

from event_model import DocumentRouter, RunRouter
import numpy
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (
    QLabel,
    QWidget,
    QVBoxLayout,
    )

from .hints import hinted_fields, guess_dimensions, extract_hints_info
import numpy as np


log = logging.getLogger('bluesky_browser')


class MPLFigureGetter:
    """
    Knows how to return existing figures or make new ones as needed
    """

    def __init__(self):
        self._figures = {}
        # Configuartion

    def get_figure(self, key, label, *args, **kwargs):
        try:
            return self._figures[key]
        except KeyError:
            return self._add_figure(key, label, *args, **kwargs)

    def _add_figure(self, key, label, *args, **kwargs):
        fig, _ = plt.subplots(*args, **kwargs)
        fig.canvas.set_window_title(label)
        self._figures[key] = fig
        return fig


class QtFigureGetter:
    """
    Knows how to return existing figures or make new ones as needed
    """
    def __init__(self, add_tab):
        self.add_tab = add_tab
        self._figures = {}

    def get_figure(self, key, label, *args, **kwargs):
        try:
            return self._figures[key]
        except KeyError:
            return self._add_figure(key, label, *args, **kwargs)

    def _add_figure(self, key, label, *args, **kwargs):
        tab = QWidget()
        fig, _ = plt.subplots(*args, **kwargs)
        canvas = FigureCanvas(fig)
        canvas.setMinimumWidth(640)
        canvas.setParent(tab)
        toolbar = NavigationToolbar(canvas, tab)
        tab_label = QLabel(label)
        tab_label.setMaximumHeight(20)

        layout = QVBoxLayout()
        layout.addWidget(tab_label)
        layout.addWidget(canvas)
        layout.addWidget(toolbar)
        tab.setLayout(layout)
        self.add_tab(tab, label)
        self._figures[key] = fig
        return fig


class QtFigureManager:
    """
    For a given Viewer, encasulate the matplotlib Figures and associated tabs.
    """
    def __init__(self, add_tab):
        # Configuartion
        self.enabled = True
        self.exclude_streams = set()
        self.figure_getter = QtFigureGetter(add_tab)

    def __call__(self, name, start_doc):
        if not self.enabled:
            return
        dimensions = start_doc.get('hints', {}).get('dimensions',
                                                    guess_dimensions(start_doc))
        log.debug('dimensions: %s', dimensions)
        line_plot_manager = LinePlotManager(self.figure_getter, dimensions)
        rr = RunRouter([line_plot_manager])
        rr('start', start_doc)
        return [rr], []


class LinePlotManager:
    """
    Manage the line plots for one FigureManager.
    """
    def __init__(self, fig_manager, dimensions):
        self.fig_manager = fig_manager
        self.start_doc = None
        self.dimensions = dimensions
        self.dim_streams = set(stream for _, stream in self.dimensions)
        if len(self.dim_streams) > 1:
            raise NotImplementedError
        # Configuration
        self.omit_single_point_plot = True

    def __call__(self, name, start_doc):
        self.start_doc = start_doc
        return [], [self.subfactory]

    def subfactory(self, name, descriptor_doc):
        if self.omit_single_point_plot and self.start_doc.get('num_points') == 1:
            return []
        if len(self.dimensions) > 1:
            return []  # This is a job for Grid.
        fields = set(hinted_fields(descriptor_doc))
        # Filter out the fields with a data type or shape that we cannot
        # represent in a line plot.
        for field in list(fields):
            dtype = descriptor_doc['data_keys'][field]['dtype']
            if dtype not in ('number', 'integer'):
                fields.discard(field)
            ndim = len(descriptor_doc['data_keys'][field]['shape'] or [])
            if ndim != 0:
                fields.discard(field)

        callbacks = []
        dim_stream, = self.dim_streams  # TODO Handle multiple dim_streams.
        if descriptor_doc.get('name') == dim_stream:
            dimension, = self.dimensions
            x_keys, stream_name = dimension
            fields -= set(x_keys)
            assert stream_name == dim_stream  # TODO Handle multiple dim_streams.
            for x_key in x_keys:
                figure_label = f'Scalars v {x_key}'
                fig = self.fig_manager.get_figure(
                    (x_key, tuple(fields)), figure_label, len(fields), sharex=True)
                for y_key, ax in zip(fields, fig.axes):

                    log.debug('plot %s against %s', y_key, x_key)

                    ylabel = y_key
                    y_units = descriptor_doc['data_keys'][y_key].get('units')
                    ax.set_ylabel(y_key)
                    if y_units:
                        ylabel += f' [{y_units}]'
                    # Set xlabel only on lowest axes, outside for loop below.

                    def func(event_page, y_key=y_key):
                        """
                        Extract x points and y points to plot out of an EventPage.

                        This will be passed to LineWithPeaks.
                        """
                        y_data = event_page['data'][y_key]
                        if x_key == 'time':
                            t0 = self.start_doc['time']
                            x_data = numpy.asarray(event_page['time']) - t0
                        elif x_key == 'seq_num':
                            x_data = event_page['seq_num']
                        else:
                            x_data = event_page['data'][x_key]
                        return x_data, y_data

                    line = Line(func, ax=ax)
                    callbacks.append(line)

                if fields:
                    # Set the xlabel on the bottom-most axis.
                    if x_key == 'time':
                        xlabel = x_key
                        x_units = 's'
                    elif x_key == 'seq_num':
                        xlabel = 'sequence number'
                        x_units = None
                    else:
                        xlabel = x_key
                        x_units = descriptor_doc['data_keys'][x_key].get('units')
                    if x_units:
                        xlabel += f' [{x_units}]'
                    ax.set_xlabel(x_key)
                    fig.tight_layout()
            # TODO Plot other streams against time.
        for callback in callbacks:
            callback('start', self.start_doc)
            callback('descriptor', descriptor_doc)
        return callbacks


def grid_factory_descriptor(all_dim_names, dim_names, shape,
                            extent, name, descriptor, origin='lower'):
    columns = hinted_fields(descriptor)
    fig, axes = find_figure(dim_names, columns)
    callbacks = []
    I_names = [c for c in columns
               if c not in all_dim_names]
    for I_name, ax in zip(I_names, axes):

        # This section defines the function for the grid callback
        def func(self, bulk_event):
            '''This functions takes in a bulk event and returns x_coords,
            y_coords, I_vals lists.
            '''
            # start by working out the scaling between grid pixels and axes
            # units
            data_range = np.array([float(np.diff(e)) for e in self.extent])
            y_step, x_step = data_range / [max(1, s - 1) for s in
                                           self.shape]
            x_min = self.extent[0]
            y_min = self.extent[2]
            # define the lists of relevant data from the bulk_event
            x_vals = bulk_event['data'][dim_names[1]]
            y_vals = bulk_event['data'][dim_names[0]]
            I_vals = bulk_event['data'][I_name]
            x_coords = []
            y_coords = []

            for x_val, y_val in zip(x_vals, y_vals):
                x_coords.append((x_val - x_min) / x_step)
                y_coords.append((y_val - y_min) / y_step)
            return x_coords, y_coords, I_vals  # lists to be returned

        grid_callback = Grid(func, shape, ax=ax,
                             extent=extent, origin=origin)
        callbacks.append(grid_callback)

    return callbacks


def grid_factory_start(name, start_doc, figure_func):
    '''
    This is a callback factory for 'grid' or 'image' plots. It takes in a
    start_doc and returns a list of callbacks that have been initialized based
    on its contents.
    '''
    shape = start_doc['shape']
    # If this isn't a 2D thing don't even bother
    if len(shape) != 2:
        return [], []
    extent = start_doc['extents']

    # define some required parameters for setting up the grid plot.
    # NOTE: THIS NEEDS WORK, in order to allow for plotting of non-grid type
    # scans the following parameters need to be passed down to here from the RE
    # This is the minimum information required to create the grid plot.

    # This section adjusts extents so that the values are centered on the grid
    # pixels
    data_range = np.array([float(np.diff(e)) for e in extent])
    y_step, x_step = data_range / [max(1, s - 1) for s in shape]
    adjusted_extent = [extent[1][0] - x_step / 2,
                       extent[1][1] + x_step / 2,
                       extent[0][0] - y_step / 2,
                       extent[0][1] + y_step / 2]

    _, dim_fields, all_dim_fields = extract_hints_info(start_doc)
    return [], [partial(grid_factory_descriptor,
                        dim_fields,
                        all_dim_fields,
                        shape,
                        adjusted_extent,)
                ]


class Line(DocumentRouter):
    """
    Draw a matplotlib Line Arist update it for each Event.

    Parameters
    ----------
    func : callable
        This must accept an EventPage and return two lists of floats
        (x points and y points). The two lists must contain an equal number of
        items, but that number is arbitrary. That is, a given document may add
        one new point to the plot, no new points, or multiple new points.
    label_template : string
        This string will be formatted with the RunStart document. Any missing
        values will be filled with '?'. If the keyword argument 'label' is
        given, this argument will be ignored.
    ax : matplotlib Axes, optional
        If None, a new Figure and Axes are created.
    **kwargs
        Passed through to :meth:`Axes.plot` to style Line object.
    """
    def __init__(self, func, *, label_template='{scan_id} [{uid:.8}]', ax=None, **kwargs):
        self.func = func
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
        self.ax = ax
        self.line, = ax.plot([], [], **kwargs)
        self.x_data = []
        self.y_data = []
        self.label_template = label_template
        self.label = kwargs.get('label')

    def start(self, doc):
        if self.label is None:
            d = collections.defaultdict(lambda: '?')
            d.update(**doc)
            label = self.label_template.format_map(d)
        else:
            label = self.label
        if label:
            self.line.set_label(label)
            self.ax.legend(loc='best')

    def event_page(self, doc):
        x, y = self.func(doc)
        self._update(x, y)

    def _update(self, x, y):
        """
        Takes in new x and y points and redraws plot if they are not empty.
        """
        if not len(x) == len(y):
            raise ValueError("User function is expected to provide the same "
                             "number of x and y points. Got {len(x)} x points "
                             "and {len(y)} y points.")
        if not x:
            # No new data. Short-circuit.
            return
        self.x_data.extend(x)
        self.y_data.extend(y)
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim(visible_only=True)
        self.ax.autoscale_view(tight=True)
        self.ax.figure.canvas.draw_idle()


class Grid(DocumentRouter):
    """
    Draw a matplotlib AxesImage Arist update it for each Event.

    The purposes of this callback is to create (on initialization) of a
    matplotlib grid image and then update it with new data for every `event`.
    NOTE: Some important parameters are fed in through **kwargs like `extent`
    which defines the axes min and max and `origin` which defines if the grid
    co-ordinates start in the bottom left or top left of the plot. For more
    info see https://matplotlib.org/tutorials/intermediate/imshow_extent.html
    or https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow

    Parameters
    ----------
    func : callable
        This must accept a BulkEvent and return three lists of floats (x
        grid co-ordinates, y grid co-ordinates and grid position intensity
        values). The three lists must contain an equal number of items, but
        that number is arbitrary. That is, a given document may add one new
        point, no new points or multiple new points to the plot.
    shape : tuple
        The (row, col) shape of the grid.
    ax : matplotlib Axes, optional.
        if ``None``, a new Figure and Axes are created.
    **kwargs
        Passed through to :meth:`Axes.imshow` to style the AxesImage object.
    """
    def __init__(self, func, shape, *, ax=None, **kwargs):
        self.func = func
        self.shape = shape
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        self.grid_data = numpy.full(self.shape, numpy.nan)
        self.image, = ax.imshow(self.grid_data, **kwargs)

    def event_page(self, doc):
        '''
        Takes in a bulk_events document and updates grid_data with the values
        returned from self.func(doc)

        Parameters
        ----------
        doc : dict
            The bulk event dictionary that contains the 'data' and 'timestamps'
            associated with the bulk event.

        Returns
        -------
        x_coords, y_coords, I_vals : Lists
            These are lists of x co-ordinate, y co-ordinate and intensity
            values arising from the bulk event.
        '''
        x_coords, y_coords, I_vals = self.func(doc)
        self._update(x_coords, y_coords, I_vals)

    def _update(self, x_coords, y_coords, I_vals):
        '''
        Updates self.grid_data with the values from the lists x_coords,
        y_coords, I_vals.

        Parameters
        ----------
        x_coords, y_coords, I_vals : Lists
            These are lists of x co-ordinate, y co-ordinate and intensity
            values arising from the event. The length of all three lists must
            be the same.
        '''

        if not len(x_coords) == len(y_coords) == len(I_vals):
            raise ValueError("User function is expected to provide the same "
                             "number of x, y and I points. Got {0} x points, "
                             "{1} y points and {2} I values."
                             "".format(len(x_coords), len(y_coords),
                                       len(I_vals)))

        if not x_coords:
            # No new data, Short-circuit.
            return

        # Update grid_data and the plot.
        self.grid_data[x_coords, y_coords] = I_vals
        self.image.set_array(self.grid_data)
