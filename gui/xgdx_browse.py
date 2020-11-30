import os
import sys
from pathlib import Path

import xgdx_browse_backend as be

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication

# Load GUI from .ui file
form = uic.loadUiType("gdx_browse.ui")


class MainWindow(form[0], form[1]):
    """
    This class create an instance of gdx_browse gui.

    """

    def __init__(self, root_dir=None):
        """
        This method initializes the gui form.

        :param root_dir:
        """
        super().__init__()
        self.setupUi(self)

        # Status variables
        self.image_set = None
        self.collection_dir = None
        self.series_len = None
        self.series_name = None
        self.series_num = None
        self.series_current_idx = 0

        # Assigning space for figures
        self.fig = None
        self.ax = None
        self.imgplot = None
        self.current_cmap = 'Gray'

        if not root_dir:
            self.dataset_dir = os.path.join(str(Path.home()), 'datasets', 'xray-imaging')
        else:
            self.dataset_dir = root_dir

        # Init ComboBox_1 with the directory names
        self.dropdown1_lock = False
        self.dropdown2_lock = False

        self.combobox_1.clear()
        self.combobox_1.addItem("<Select a dataset>")
        self.combobox_1.addItems(os.listdir(self.dataset_dir))
        self.dropdown1_lock = True

        # Init combo_box_3 using colormaps names.
        self.combobox_3.clear()
        self.combobox_3.addItems(list(be.cmaps.keys()))

        # Init checkbox
        self.check_box_1.setChecked(False)

        # Setup signals

        # Comboboxes
        self.combobox_1.currentIndexChanged.connect(self.on_combobox_1_changed)
        #self.combobox_2.currentTextChanged.connect(self.combobox_2.setCurrentIndex)
        self.combobox_2.currentIndexChanged.connect(self.on_combobox_2_changed)
        self.combobox_3.currentIndexChanged.connect(self.on_combobox_3_changed)

        # Buttons
        self.push_button_1.clicked.connect(self.on_pb_clicked)
        self.push_button_2.clicked.connect(self.on_pb_clicked)

        # CheckBoxes
        self.check_box_1.stateChanged.connect(self.cb1_state_changed)

    def cb1_state_changed(self):
        """
        Check state of checkbox_1
        """
        if self.check_box_1.isChecked():
            self.fig, self.ax = be.show_ground_truth(self.image_set, self.series_name,
                                                     self.series_current_idx, self.fig, self.ax)
        else:
            # Load image
            img = be.load_image(self.image_set, self.series_name, self.series_current_idx)
            msgs = be.compose_image_message(self)
            self.fig, self.ax, self.imgplot = be.show_image(img=img, messages=msgs, fig=self.fig, ax=self.ax,
                                                            cmap=self.current_cmap)

    def on_pb_clicked(self):
        """
        Check whether buttons were clicked.
        """
        sending_button = self.sender()

        # Check the clicked button
        if sending_button.objectName() == 'push_button_1':
            if be.check_first_image(self.series_current_idx):
                self.series_current_idx -= 1
        elif sending_button.objectName() == 'push_button_2':
            if be.check_last_image(self.series_current_idx, self.series_len):
                self.series_current_idx += 1

        # Load next image
        img = be.load_image(self.image_set, self.series_name, self.series_current_idx)
        #msgs = be.compose_image_message(self.image_set, self.series_name, self.series_current_idx)
        msgs = be.compose_image_message(self)

        self.fig, self.ax, self.imgplot = be.show_image(
            img=img,
            messages=msgs,
            fig=self.fig,
            ax=self.ax,
            cmap=self.current_cmap)

        if self.check_box_1.isChecked():
            self.fig, self.ax = be.show_ground_truth(self.image_set, self.series_name,
                                                     self.series_current_idx, self.fig, self.ax)

    def on_combobox_1_changed(self):
        """
        Load an image-set for the selected collection.
        """

        if self.dropdown1_lock:
            self.dropdown2_lock = False
            self.image_set = be.load_dataset(self.combobox_1.currentText())  # Load image set

            # Update Combobox2 items
            self.combobox_2.clear()
            self.combobox_2.addItem('<Select a series>')
            self.combobox_2.addItems(be.get_dir_list(self.image_set.dataset_path))
            self.dropdown2_lock = True

    def on_combobox_2_changed(self):
        """
        This method control changes on the combobox to select image series.
        """

        if self.dropdown2_lock:

            # Initialize figure
            if self.fig:
                be.close_figure(self.fig)
                self.fig = None
                self.ax = None
                self.imgplot = None
                self.check_box_1.setChecked(False)

            self.series_name = self.combobox_2.currentText()
            self.series_num = self.image_set.describe()['series']
            self.series_len = be.get_series_length(self.image_set.dataset_path, self.series_name)
            self.series_current_idx = 1

            img = be.load_image(self.image_set, self.series_name, self.series_current_idx)
            msgs = be.compose_image_message(self)  #.image_set, self.series_name, self.series_current_idx)

            self.fig, self.ax, self.imgplot = be.show_image(
                img=img,
                messages=msgs,
                fig=self.fig,
                ax=self.ax,
                cmap=self.current_cmap)

    def on_combobox_3_changed(self):
        try:
            self.current_cmap = self.combobox_3.currentText()
            self.imgplot.set_cmap(be.cmaps[self.current_cmap])
            self.fig.canvas.draw()
            self.fig.show()
        except Exception as err:
            print('Error: '.format(err))


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
