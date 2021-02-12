from typing import Union
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread, QEvent
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMainWindow, QTableView, QHeaderView, QAction, QFileDialog, QMenuBar, QAction, QMenu, QStatusBar, QProgressBar, QLabel, QSizePolicy, QTreeView

import pandas as pd
import os, re, json, sys, logging
import pathlib, shutil

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../RSAtrace3D'))
    logging.basicConfig(level=logging.INFO)

from mod.Extensions.__backbone__ import ExtensionBackbone
from DATA import RSA_Vector
from GUI import QtMain

class Extension_RSATrace4D(QMainWindow, ExtensionBackbone):
    built_in = True
    label = 'RSA Trace4D'
    status_tip = 'Pop a RSAtrace4D window up.'
    index = 0
    version = 1

    def __init__(self, parent: Union[QtMain, None]):
        super().__init__(parent=parent)
        self.__parent = parent
        self.setWindowTitle('RSA Trace4D')
        self.resize(600,400)
        self.treeview = SequentialTreeView(parent=self)
        self.setCentralWidget(self.treeview)
        self.setAcceptDrops(True)

        self.menubar = QtMenubar(parent=self)
        self.setMenuBar(self.menubar)
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.WindowActivate:
            self.treeview.repaint()
            return True
        else:
            return super().eventFilter(obj, event)


    def parent(self):
        return self.__parent

    def dragEnterEvent(self, ev):
        ev.accept()
        
    def dropEvent(self, ev):
        ev.accept()
        dir_list = [u.toLocalFile() for u in ev.mimeData().urls()]
        dir_list = [d for d in dir_list if os.path.isdir(d)]
        if len(dir_list) != 1:
            return

        self.treeview.import_from(directory=dir_list[0])
 
    def save_all(self):
        item_count = self.treeview.model.rowCount()

        for i in range(self.treeview.model.rowCount()):
            base_item = self.treeview.model.item(i, 0)
            base_path = pathlib.Path(base_item.text())
            rsainfo_file = str(base_path)+'.rsainfo'

            child_count = base_item.rowCount()
            for cc in range(child_count):
                child_item = base_item.child(cc, 0)
                child_path = pathlib.Path(child_item.text())
                relative_child_path = str(child_path.relative_to(base_path))

                child_item = base_item.child(cc, 1)
                series_value = int(child_item.text())
                print(relative_child_path, series_value)
            
class QtStatusBarW(QObject):
    pyqtSignal_update_progressbar = pyqtSignal(int, int, str)

    def __init__(self, parent):
        super().__init__()
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.start()

        self.widget = QtStatusBar(parent=parent)
        self.pyqtSignal_update_progressbar.connect(self.widget.update_progress)

    def set_main_message(self, msg):
        self.widget.set_main_message(msg=msg)

class QtStatusBar(QStatusBar):
    def __init__(self, parent):
        super().__init__(**{'parent': parent})

        self.progress = QProgressBar()
        self.status_msg = QLabel('')

        self.addWidget(self.progress)
        self.addWidget(self.status_msg, 2048)

        self.progress.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding))
        self.progress.setValue(0)
        
    def update_progress(self, i, maximum, msg):
        self.progress.setMaximum(maximum)
        self.progress.setValue(i +1)

        self.set_main_message(f'{msg}: {i+1} / {maximum}')

    def set_main_message(self, msg):
        if self.parent().is_control_locked():
            msg = 'BUSY: '+msg 
        else:
             msg = 'READY: '+msg 

        self.status_msg.setText(msg)

class QtMenubar(QMenuBar):
    def __init__(self, parent):
        super().__init__(**{'parent': parent})
        self.__parent = parent
        self.build()

    def build(self):
        self.menu_file = QMenu('&File')
        self.addMenu(self.menu_file)

        #// file menu
        self.act_open_volume = QAction(
            text='Open volume', parent=self.parent(), shortcut='Ctrl+O',
            statusTip='Open volume', triggered=self.on_act_open_volume)
        self.menu_file.addAction(self.act_open_volume)

        self.act_save = QAction(
            text='Save sequence file', 
            triggered=self.on_act_save)
        self.menu_file.addAction(self.act_save)

        self.menu_file.addSeparator()
        self.act_exit = QAction(
            text='Exit', parent=self.parent(), shortcut='Ctrl+Q',
            statusTip='Exit application', triggered=self.parent().close)
        self.menu_file.addAction(self.act_exit)

    def update(self):
        for m, item in self.__dict__.items():
            if m.lower().startswith(('act_', 'menu_')):
                item.setEnabled(self.parent().is_control_locked()==False)

        if self.parent().is_control_locked():
            return

        for m, item in self.__dict__.items():
            if m.lower().startswith(('act_', 'menu_')):
                if hasattr(item, 'auto_enable_volume'):
                    if item.auto_enable_volume == True:
                        item.setEnabled(self.RSA_components().volume.is_empty()==False)

    def on_act_open_volume(self):
        directory = QFileDialog.getExistingDirectory(self, 'Volume directory', os.path.expanduser('~'))
        if directory == '':
            return False
       
        self.parent().load_from(directory=directory)

    def on_act_save(self):
        self.__parent.save_all()

    def on_menu_history(self):
        obj  = QObject.sender(self)
        name = obj.data()

        self.parent().load_from(directory=name)

    def on_act_close_volume(self):
        self.parent().close_volume()

    def on_act_export_root_csv(self):
        df = self.parent().tableview.to_pandas_df()
        print(df)
        return
        df = self.GUI_components().treeview.to_pandas_df()
        csv_fname = self.RSA_components().file.root_traits_file
        volume_name = self.RSA_components().vector.annotations.volume_name()
        with open(csv_fname, 'w', newline="") as f:
            f.write(f'# This file is a summary of root traits measured by RSAtrace3D.\n')
            f.write(f'# Volume name: {volume_name}\n')
            df.to_csv(f)

    def on_menu_interpolation(self):
        obj  = QObject.sender(self)
        name = obj.data()
        self.RSA_components().vector.annotations.set_interpolation(interpolation=name)

    def on_menu_extensions(self):
        obj  = QObject.sender(self)
        name = obj.data()

        self.parent().extensions.activate_window(label=name)


class SequentialTreeViewModel(QStandardItemModel):
    def __init__(self):
        super().__init__()
        self.header_labels = ['directory', 'series', 'rinfo']
        self.setHorizontalHeaderLabels(self.header_labels)

    def header_index(self, label):
        return self.header_labels.index(label) if label in self.header_labels else None

    def directory(self, index):
        item = self.itemFromIndex(index)
        nrow = item.row()
        parent = item.parent()

        if parent is None:
            target_index = self.index(nrow, self.header_index(label='directory'))
            target_item = self.itemFromIndex(target_index)
        else:
            target_item = parent.child(nrow, self.header_index(label='directory'))
        
        return target_item.data()

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            if index.column() == self.header_index(label='directory'):
                path = self.itemFromIndex(index).text()
                return os.path.basename(path)

            elif index.column() == self.header_index(label='rinfo'):
                return 'OK' if self.is_rinfo_avairable(index=index) else ''

        return super().data(index, role)

    def rinfo_path(self, index):
        directory = self.directory(index=index)
        return f'{directory}.rinfo'

    def is_rinfo_avairable(self, index):
        rinfo_path = self.rinfo_path(index=index)
        if os.path.isfile(rinfo_path):
            return True
        else:
            return False

class SequentialTreeView(QTreeView):
    def __init__(self, parent: Extension_RSATrace4D):
        super().__init__(**{'parent': parent})
        self.__parent = parent
        self.model = SequentialTreeViewModel()
        self.setModel(self.model)

    def parent(self):
        return self.__parent

    #// key press event
    def keyPressEvent(self, ev):
        ev.accept()
        main = self.parent().parent()

        if main.is_control_locked():
            return 

        if ev.key() == Qt.Key_Return and not ev.isAutoRepeat():
            selection_model = self.selectionModel()

            if not selection_model.hasSelection():
                return
            
            index = selection_model.selectedRows(0)[0]
            selected_item = self.model.itemFromIndex(index)
            parent_item = selected_item.parent()
            if parent_item is None:
                return

            trace_dict = {}
            row = index.row()
            if row != 0 and not self.model.is_rinfo_avairable(index=index):
                for i in range(row):
                    target_index = parent_item.child(row-i-1, 0).index()
                    if self.model.is_rinfo_avairable(index=target_index):
                        rinfo_path = self.model.rinfo_path(target_index)
                        with open(rinfo_path, 'r') as f:
                            trace_dict = json.load(f)
                        break
                
                trace_dict['#annotations']['volume name'] = os.path.basename(selected_item.text())
                

            main.load_from(directory=selected_item.text(), rinfo_dict=trace_dict)

    def import_from(self, directory):
        row = [QStandardItem(directory)]
        row[0].setData(directory)
        base_item = row[0]

        child_dir_list = os.listdir(directory)
        regex = re.compile(r'\d+')
        series_list = [regex.findall(d) for d in child_dir_list]
        series_list = [int(s[-1]) if len(s) != 0 else 0 for s in series_list]

        list_ = list(zip(*[child_dir_list, series_list]))
        list_.sort(key=lambda x: x[1])
        list_ = list(zip(*list_))

        for d, series in zip(list_[0], list_[1]):
            child_dir = os.path.join(directory, d)
            if os.path.isdir(child_dir):
                child_row = [QStandardItem(child_dir), QStandardItem(f'{series}'), QStandardItem('')]
                child_row[0].setData(child_dir)
                child_row[1].setData(series)
                base_item.appendRow(child_row)

        self.model.appendRow(row)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    ins = Extension_RSATrace4D(parent=None)
    ins.show()
    app.exec_()