import argparse
from datetime import datetime
import logging
import sys
import time
from . import __version__

from qtpy.QtCore import QDateTime
from qtpy.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout)
from .search import SearchWidget, SearchState
from .summary import SummaryWidget
from .viewer import Viewer
from .zmq import ConsumerThread


log = logging.getLogger('bluesky_browser')


class CentralWidget(QWidget):
    """
    Encapsulates all widgets and models. Connect signals on __init__.
    """
    def __init__(self, *args,
                 catalog, search_result_row, menuBar,
                 zmq_address=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Define models.
        search_state = SearchState(
            catalog=catalog,
            search_result_row=search_result_row)

        # Define widgets.
        self.search_widget = SearchWidget()
        self.summary_widget = SummaryWidget()

        left_pane = QVBoxLayout()
        left_pane.addWidget(self.search_widget)
        left_pane.addWidget(self.summary_widget)

        right_pane = QVBoxLayout()
        self.viewer = Viewer(menuBar=menuBar)

        right_pane.addWidget(self.viewer)

        layout = QHBoxLayout()
        layout.addLayout(left_pane)
        layout.addLayout(right_pane)
        self.setLayout(layout)

        def show_double_clicked_entry(index):
            search_state.search_results_model.emit_open_entries(None, [index])

        # Set models, connect signals, and set initial values.
        now = time.time()
        ONE_WEEK = 60 * 60 * 24 * 7
        self.search_widget.search_results_widget.setModel(
            search_state.search_results_model)
        self.search_widget.search_input_widget.search_bar.textChanged.connect(
            search_state.search_results_model.on_search_text_changed)
        self.search_widget.catalog_selection_widget.catalog_list.setModel(
            search_state.catalog_selection_model)
        self.search_widget.search_input_widget.until_widget.dateTimeChanged.connect(
            search_state.search_results_model.on_until_time_changed)
        self.search_widget.search_input_widget.until_widget.setDateTime(
            QDateTime.fromSecsSinceEpoch(now + ONE_WEEK))
        self.search_widget.search_input_widget.since_widget.dateTimeChanged.connect(
            search_state.search_results_model.on_since_time_changed)
        self.search_widget.search_input_widget.since_widget.setDateTime(
            QDateTime.fromSecsSinceEpoch(now - ONE_WEEK))
        self.search_widget.catalog_selection_widget.catalog_list.currentIndexChanged.connect(
            search_state.set_selected_catalog)
        self.search_widget.search_results_widget.selectionModel().selectionChanged.connect(
            search_state.search_results_model.emit_selected_result)
        self.search_widget.search_results_widget.doubleClicked.connect(
            show_double_clicked_entry)
        search_state.search_results_model.selected_result.connect(
            self.summary_widget.set_entries)
        search_state.search_results_model.open_entries.connect(
            self.viewer.show_entries)
        self.summary_widget.open.connect(self.viewer.show_entries)
        self.viewer.tab_titles.connect(self.summary_widget.cache_tab_titles)
        search_state.search_results_model.valid_custom_query.connect(
            self.search_widget.search_input_widget.mark_custom_query)
        search_state.enabled = True
        search_state.search()

        if zmq_address:
            self.consumer_thread = ConsumerThread(zmq_address=zmq_address)
            self.consumer_thread.documents.connect(self.viewer.consumer)
            self.consumer_thread.start()


def main():
    parser = argparse.ArgumentParser(description='Prototype bluesky data browser',
                                     epilog=f'version {__version__}')
    parser.register('action', 'demo', _DemoAction)
    parser.add_argument('catalog', type=str)
    parser.add_argument('-z', '--zmq-address', dest='zmq_address',
                        default=None, type=str,
                        help='0MQ remote dispatcher address (host:port)')
    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--demo', action='demo',
                        default=argparse.SUPPRESS,
                        help="Launch the app with example data.")
    args = parser.parse_args()
    if args.verbose:
        handler = logging.StreamHandler()
        handler.setLevel('DEBUG')
        log.addHandler(handler)
        log.setLevel('DEBUG')
    app = build_app(args.catalog, zmq_address=args.zmq_address)
    sys.exit(app.exec_())


def build_app(catalog_uri, zmq_address=None):
    from intake import Catalog
    catalog = Catalog(catalog_uri)

    # TODO Make search_result_row configurable.

    def search_result_row(entry):
        start = entry.metadata['start']
        stop = entry.metadata['stop']
        start_time = datetime.fromtimestamp(start['time'])
        duration = datetime.fromtimestamp(stop['time']) - start_time
        str_duration = str(duration)
        return {'Unique ID': start['uid'][:8],
                'Transient Scan ID': str(start.get('scan_id', '-')),
                'Plan Name': start.get('plan_name', '-'),
                'Start Time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Duration': str_duration[:str_duration.index('.')],
                'Exit Status': '-' if stop is None else stop['exit_status']}

    app = QApplication([b'Bluesky Browser'])
    app.main_window = QMainWindow()
    central_widget = CentralWidget(
        catalog=catalog,
        zmq_address=zmq_address,
        search_result_row=search_result_row,
        menuBar=app.main_window.menuBar)
    app.main_window.setCentralWidget(central_widget)
    app.main_window.show()
    return app


class _DemoAction(argparse.Action):
    """
    A special action that generates example data and launches the app.

    This overrides the parser's required arguments the same way that --help
    does, so that the user does not have to pass in a catalog in this case.
    """
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        handler = logging.StreamHandler()
        handler.setLevel('DEBUG')
        log.addHandler(handler)
        log.setLevel('DEBUG')

        from .demo import generate_example_catalog, stream_example_data
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            catalog_filepath = generate_example_catalog(directory)
            zmq_address, proxy_process, publisher_process = stream_example_data(directory)
            app = build_app(catalog_filepath, zmq_address)
            app.main_window.centralWidget().viewer.off.setChecked(True)
            try:
                ret = app.exec_()
            finally:
                proxy_process.terminate()
                publisher_process.terminate()
                sys.exit(ret)
                parser.exit()


if __name__ == '__main__':
    main()
