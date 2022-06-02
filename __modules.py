
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from glob import glob
from types import FunctionType
from typing import Any, Dict, Final, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from more_itertools import pairwise
from skimage import io
from tensorflow.python.keras.engine.functional import Functional

import __common
from __common import tqdm
from __model import RootClassificationModel
from __rinfo import ID_Object, RSA_Vector

warnings.simplefilter('ignore')
tf.get_logger().setLevel('ERROR')
logger = __common.logger


@dataclass(frozen=True)
class VolumeSeries(object):
    root_directory: str

    def __post_init__(self):
        assert os.path.isdir(self.root_directory)
        
    @property
    def volume_list(self):
        __volume_list = sorted(glob(self.root_directory+'/**/'))
        __volume_list = [source for source in __volume_list if VolumeLoader(source).is_volume_directory()]
        return __volume_list

    @property
    def volume_number(self):
        return len(self.volume_list)

    def does_contain_volumes(self):
        return self.volume_number > 0

@dataclass(frozen=True)
class RSAtrace4D_Series(VolumeSeries):

    @property
    def rinfo_file_list(self):
        return sorted(glob(self.root_directory+'/*.rinfo'))

    @property
    def rinfo_file_number(self):
        return len(self.rinfo_file_list)

    def is_valid(self):
        if self.rinfo_file_number != 1 or self.volume_number < 2:
            return False

        if os.path.basename(self.rinfo_file_list[0])[:-6] == os.path.basename(self.volume_list[-1][:-1]):
            return True

        return False

    @property
    def model_destination(self):
        return os.path.join(self.root_directory, '.models_for_BP')

@dataclass(frozen=True)
class VolumeLoader(object):
    source_directory: str
    minimum_file_number: int = 64
    extensions: Tuple = ('.cb', '.png', '.tif', '.tiff', '.jpg', '.jpeg')

    def __post_init__(self):
        assert os.path.isdir(self.source_directory)

    @property
    def image_file_list(self):
        img_files = [os.path.join(self.source_directory, f) for f in os.listdir(self.source_directory)]

        ext_count = []
        for ext in self.extensions:
            ext_count.append(len([f for f in img_files if f.lower().endswith(ext)]))

        target_extension = self.extensions[ext_count.index(max(ext_count))]
        return sorted([f for f in img_files if f.lower().endswith(target_extension)])

    @property
    def image_file_number(self):
        return len(self.image_file_list)

    def is_volume_directory(self):
        return self.image_file_number >= self.minimum_file_number

    def load(self) -> np.ndarray:
        assert os.path.isdir(self.source_directory)

        logger.info(f'Loading {self.image_file_number} image files: {self.source_directory}')

        ndarray = np.array(
            [io.imread(f) for f in tqdm(self.image_file_list)]
        )

        return ndarray

@dataclass
class RootData(object):
    ID_string: ID_Object
    coordinates: np.ndarray
    trimmed_data: np.ndarray
    label: int

    def get_train_data(self, number:int = -1) -> Tuple[np.ndarray, np.ndarray]:
        x = self.trimmed_data
        y = np.array(np.array(np.ones(len(self.trimmed_data))*self.label))
        if number < 0:
            return (x, y)
        else:
            index = np.random.choice(len(self.trimmed_data), number)
            return (x[index], y[index])

@dataclass
class RootDataManager(object):
    root_data_dict: Dict[ID_Object, RootData] = field(init=False, default_factory=dict)


    def append(self, root_data: RootData):
        self.root_data_dict.update({root_data.ID_string: root_data})

    def get_all_coordinates(self) -> np.ndarray:
        coordinates = [v.coordinates for v in self.root_data_dict.values()]
        coordinates = np.array(np.unique(np.concatenate(coordinates), axis=0))
        return coordinates

@dataclass
class DataMaker(object):
    source_dir:str
    rinfo_file:str
    root_data_manager: RootDataManager = field(init=False, default_factory=RootDataManager)
    trim_radius: int = field(init=False, default=8)

    def __post_init__(self):
        assert os.path.isdir(self.source_dir)
        assert os.path.isfile(self.rinfo_file)

        self.rsa_vector = RSA_Vector()
        with open(self.rinfo_file, 'r') as f:
            trace_dict = json.load(f)

        self.rsa_vector.load_from_dict(trace_dict=trace_dict, file=self.rinfo_file)
        self.volume = VolumeLoader(source_directory=self.source_dir).load()

    def get_ID_string_list(self):
        return [ID_string for ID_string, _ in self.root_data_manager.root_data_dict.items() if ID_string != '00-00-00']

    def load_segments(self):
        for base_node in self.rsa_vector:
            for root_node in base_node:
                ID_string = root_node.ID_string()
                coordinates = np.array(root_node.interpolated_polyline())
                coordinates = np.array(np.unique(coordinates, axis=0))
                trimmed_data = np.array(
                    [Volume3D(self.volume).get_trimmed_volume(co, self.trim_radius) for co in coordinates]
                )
                trimmed_data = np.array(trimmed_data.clip(0,255), dtype=np.uint8)
                root_data = RootData(
                    ID_string=ID_string,
                    coordinates=coordinates,
                    trimmed_data=trimmed_data,
                    label=1
                )
                self.root_data_manager.append(
                    root_data
                )

        positive_co_list = self.root_data_manager.get_all_coordinates()
        negative_co_list = []
        
        while(len(negative_co_list) != len(positive_co_list)):
            co = [np.random.randint(0, self.volume.shape[d]-1) for d in range(3)]
            radius = np.linalg.norm([self.volume.shape[1]//2-co[1], self.volume.shape[2]//2-co[2]])
            if radius > (self.volume.shape[1]+1)//2:
                continue

            distance = ((positive_co_list-np.array(co))**2).sum(axis=1).min()
            if distance <= self.trim_radius**2:
                continue

            negative_co_list.append(co)

        negative_co_list = np.array(negative_co_list)

        trimmed_data = np.array([
            Volume3D(self.volume).get_trimmed_volume(
                center=co, 
                radius=self.trim_radius
            ) for co in negative_co_list 
        ])
        trimmed_data = np.array(trimmed_data.clip(0,255), dtype=np.uint8)

        root_data = RootData(
            ID_string=ID_Object("00-00-00"), 
            coordinates=negative_co_list, 
            trimmed_data=trimmed_data, 
            label=0
        )
        self.root_data_manager.append(root_data=root_data)

        logger.info(f'Positive segments: {positive_co_list.shape}')
        logger.info(f'Negative segments: {negative_co_list.shape}')

    def __getitem__(self, ID_string: ID_Object):
        return self.root_data_manager.root_data_dict[ID_string]

    def get_all_train_data_set(self):
        x, y = zip(*[v.get_train_data() for v in self.root_data_manager.root_data_dict.values()])
        x = np.concatenate(x)
        y = np.concatenate(y)
        return (np.array(x), np.array(y))

    def get_train_data_set(self, ID_string: ID_Object):
        positive_root_data = self[ID_string]
        negative_root_data = self[ID_Object('00-00-00')]

        positive_X, positive_Y = positive_root_data.get_train_data()
        negative_X, negative_Y = negative_root_data.get_train_data(number=len(positive_X))

        x_train = np.concatenate([positive_X, negative_X])
        y_train = np.concatenate([positive_Y, negative_Y])

        return (x_train, y_train)

@dataclass
class Volume3D(object):
    ndarray: np.ndarray

    def get_trimmed_volume(self, center, radius):
        assert self.ndarray is not None

        S = radius*2+1
        pos = [int(i) for i in center]

        #// slices for cropping
        slices = []
        slice_indented = []
        for d in range(3):
            slices.append(slice(max(pos[d]-radius, 0), min(pos[d]+radius+1, self.ndarray.shape[d])))
            indent = -min(pos[d]-radius, 0)
            slice_indented.append(slice(indent, slices[d].stop-slices[d].start+indent))

        cropped = np.zeros((S, S, S), dtype=np.uint16)
        cropped[tuple(slice_indented)] = self.ndarray[tuple(slices)]

        return cropped

class ThresholdEarlyStopping(Callback):
    def __init__(self, monitor='loss', patience=0):
        self.monitor = monitor
        self.patience = patience
        self.wait = 0
        super().__init__()

    def on_train_begin(self, logs={}):
        self.wait = 0

    def on_epoch_end(self, batch, logs={}):
        loss = logs.get(self.monitor)
        if loss < 0.05:
            self.wait += 1
        else:
            self.wait = 0
            
        if self.wait >= self.patience:
            assert self.model is not None
            self.model.stop_training = True

@dataclass
class RootClassModel(object):
    model: Functional = field(repr=False)
    rescale_factor: float = field(init=False, default=1./255)

    def get_datagen_for_train(self):
        return ImageDataGenerator(
            horizontal_flip=True, 
            vertical_flip=True,
            rescale=self.rescale_factor,
        )

    def get_datagen_for_predict(self):
        return ImageDataGenerator(
            rescale=self.rescale_factor,
        )

    def fit(self, X_list: np.ndarray, Y_list: np.ndarray, epochs: int=100, patience: int=5):
        datagen = self.get_datagen_for_train()
        self.history = self.model.fit_generator(
            datagen.flow(X_list, Y_list, batch_size=32),
            steps_per_epoch=len(X_list) / 32, 
            epochs=epochs, 
            callbacks=[ThresholdEarlyStopping(patience=patience)],
        )
        return self.history

    def predict(self, x):
        datagen = self.get_datagen_for_predict()
        return self.model.predict(datagen.flow(x), verbose=1)

    def get(self):
        return self.model

    def save_weights(self, *args, **kwargs):
        self.model.save_weights(*args, **kwargs)

@dataclass
class ModelTraining(object):
    model_directory: str
    core_model_name: str = field(init=False, default='core_model.hdf5')

    def __post_init__(self):
        os.makedirs(self.model_directory, exist_ok=True)

    def core_model_training(self, data_maker: DataMaker):
        logger.info(f'Core model training.')

        core_model_path = os.path.join(self.model_directory, self.core_model_name)
        if os.path.isfile(core_model_path):
            logger.info(f'[skip] Core model found: {core_model_path}')
        else:
            logger.info(f'Starting core model training.')
            root_classification_model = RootClassificationModel()
            core_model = root_classification_model.get()
            core_model.summary()

            x_train, y_train = data_maker.get_all_train_data_set()

            root_class_model = RootClassModel(model=core_model)
            root_class_model.fit(x_train, y_train, epochs=100, patience=5)
            root_class_model.save_weights(core_model_path)
            logger.info(f'Core model saved: {core_model_path}')

    def individual_model_training(self, data_maker: DataMaker): 
        logger.info(f'Individual model training.')

        core_model_path = os.path.join(self.model_directory, self.core_model_name)
        for ID_string in data_maker.get_ID_string_list():
            model_path = os.path.join(self.model_directory, ID_string)+".hdf5"
            if os.path.isfile(model_path):
                logger.info(f'[skip] Individual model found: {model_path}')
                continue
            else:
                logger.info(f'Starting core model training: {ID_string}')

                x_train, y_train = data_maker.get_train_data_set(ID_string=ID_string)

                model = RootClassificationModel(pretrained_weights=core_model_path).get()
                root_class_model = RootClassModel(model=model)

                root_class_model.fit(np.array(x_train), np.array(y_train), epochs=500, patience=5)
                root_class_model.save_weights(model_path)
                logger.info(f'Individual model saved: {model_path}')

@dataclass
class ModelPredicting(object):
    dir_list: List[str]
    model_dir_path: str
    rsa_vector: RSA_Vector

    def __post_init__(self):
        self.result_figure_dir_path = os.path.join(self.model_dir_path, '.result_figures')
        self.model_dict: Dict[ID_Object, RootClassificationModel] = {}
        for base_node in self.rsa_vector:
            for root_node in base_node:
                ID_string = root_node.ID_string()
                model_path = os.path.join(self.model_dir_path, ID_string)+'.hdf5'
                model = RootClassificationModel(model_path)

                self.model_dict.update(
                    {ID_string: model}
                )

    def proceed_single(self, indir:str):
        volume = VolumeLoader(source_directory=indir).load()
        regex = re.compile('\d+') #type:ignore
        numbers = regex.findall(indir)

        if len(numbers) == 0:
            error_msg = f'The directory name must contain a number representing the time series.'
            logger.error(error_msg)
            raise Exception(error_msg)

        time_series_number: Final[int] = int(numbers[-1])


        logger.info(f'Performing backward prediction: {indir}')

        list_label = []
        list_number = []
        list_index = []
        for base_node in self.rsa_vector:
            for root_node in base_node:
                ID_string = root_node.ID_string()
                polyline = root_node.interpolated_polyline()
                polyline = np.unique(polyline, axis=0)

                seg_list = []
                for co in polyline:
                    trimmed = Volume3D(volume).get_trimmed_volume(center=co, radius=8)
                    trimmed = np.array(trimmed.clip(0,255), dtype=np.uint8)
                    seg_list.append(trimmed)

                seg_list = np.array(seg_list, dtype='float32')
                seg_list /= 255

                res = self.model_dict[ID_string].get().predict_generator(seg_list)
                res = [i[0] for i in res][::-1]

                data = res
                r = []
                for i in range(len(data)):
                    if i == 0:
                        m = np.ones(len(data), dtype='float32')    
                    elif i < len(data)-1:
                        m = np.concatenate([np.zeros(i, dtype='float32'), np.ones(len(data)-i, dtype='float32')])
                    else:
                        m = np.zeros(len(data), dtype='float32')
                    m = np.array(m)
                    s = ((data-m)**2).sum()
                    r.append(s)

                r = np.array(r)
                index = int(r.argmin())

                fig_outdir = os.path.join(self.result_figure_dir_path, ID_string)

                plt.cla()
                plt.figure(figsize=(6, 3), dpi=300)
                plt.subplots_adjust(bottom=0.2)
                plt.ylabel('score')
                plt.xlabel('index from the root tip')
                plt.title(f'DAS {time_series_number}')
                plt.plot(res, label=os.path.basename(indir))
                plt.axvline(x=index, c='red')
                plt.ylim([-0.1,1.1])
                
                os.makedirs(fig_outdir, exist_ok=True)

                plt.savefig(
                    os.path.join(fig_outdir, f'{time_series_number:02}.png'),
                    format="png", dpi=300)

                list_label.append(ID_string.rootID())
                list_number.append(len(r))
                list_index.append(index)

        return (list_label, list_number, list_index)

    
    def proceed_all(self):
        flag = True
        df_list = []
        labels = []
        for d in sorted(self.dir_list, reverse=True):
            list_label, list_number, list_index = self.proceed_single(indir=d)
            if flag:
                df_list.append(list_label)
                labels.append('label')
                df_list.append(list_number)
                labels.append('number_of_node')
                flag = False
            df_list.append(list_index)

            regex = re.compile('\d+') #type:ignore
            numbers = regex.findall(d)
            number = numbers[-1]
            labels.append(int(number))

        df = pd.DataFrame(df_list, index=labels)
        df.transpose()

        return df

@dataclass
class SingleRoot(object):
    root_id: int
    das_list: List[int] = field(repr=False)
    maximum_node_count: int
    index_list: np.ndarray = field(repr=False)
    relative_position_list: np.ndarray = field(init=False, repr=False)
    fitting_params: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.relative_position_list = np.array(self.index_list/self.maximum_node_count)

    def make_plotting_array(self, 
            start_x: int, 
            end_x: int, 
            y_clip: float, 
            das_list: List[int] = None #type: ignore
            ) -> Tuple[np.ndarray, np.ndarray]:
        das_list = das_list or self.das_list

        def fun(x1, x2, y1, y2, x):
            return (y2-y1)/(x2-x1)*(x-x1)+y1

        res = [fun(end_x, start_x, 0, 1, d) for d in das_list]
        res = np.array(res).clip(0,y_clip)
        return np.array(das_list), np.array(res)

    def __ascertain_fitting_params(func: FunctionType):#type:ignore
        def inner(self: 'SingleRoot', *args, **kwargs):
            if len(self.fitting_params) == 0:
                self.__fit()
            return func(self, *args, **kwargs)
        return inner

    @__ascertain_fitting_params #type:ignore
    def get_fitting_curve(self, das_list=None, **additional_params):
        params = self.fitting_params.copy()
        params.update(additional_params)
        return self.make_plotting_array(**params, das_list=das_list)

    @__ascertain_fitting_params #type:ignore
    def get_root_params(self):
        return_dict = {
            'root_id': self.root_id,
            'start_of_elongation': self.fitting_params['start_x'],
            'end_of_elongation': self.fitting_params['end_x'],
            'elongation_period': self.fitting_params['end_x']-self.fitting_params['start_x'] if self.fitting_params['y_clip']!=0 else 0
        }

        return return_dict


    def __fit(self):
        params = []

        n = len(self.relative_position_list)
        y_threshold = 1
        for y in range(2, 9):
            y /= 10
            
            a = np.array(self.relative_position_list-y)**2
            a = sum(a<0.1**2)/n

            if a > 1/3:
                y_threshold = y+0.1
                break

        threshold_das = min(self.das_list)-1
        for i, y in enumerate(self.relative_position_list): #type:ignore
            if y > y_threshold:
                threshold_das = self.das_list[i]
                break

        mask = np.array(self.das_list)>threshold_das

        min_score = np.inf
        for start_x in range(0, max(self.das_list)+1):
            for end_x in self.das_list:
                if start_x >= end_x:
                    continue

                for y_clip in range(0, 11):
                    y_clip /= 10

                    _, y_array = self.make_plotting_array(start_x, end_x, y_clip, self.das_list)
                    loss = (self.relative_position_list-y_array)[mask]
                    score = np.sum(loss**2)
                    if score < min_score:
                        min_score = score
                        self.fitting_params = {
                            'start_x': start_x,
                            'end_x': end_x,
                            'y_clip': y_clip,
                        }
        return

@dataclass
class RSA(object):
    single_roots: List[SingleRoot] = field(init=False, default_factory=list, repr=False)
    single_root_number: int = field(init=False, default=0)

    def append(self, single_root: SingleRoot):
        self.single_roots.append(single_root)
        self.single_root_number = len(self.single_roots)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i == len(self.single_roots):
            raise StopIteration()
        value = self.single_roots[self._i]
        self._i += 1
        return value

    def __len__(self):
        return self.single_root_number

    def get_root_id_list(self):
        return [single_root.root_id for single_root in self.single_roots]

@dataclass
class Fitting(object):
    source_csv: str = field(repr=False)
    RSA: RSA = field(init=False, default_factory=RSA) #type: ignore

    def __post_init__(self):
        df = pd.read_csv(self.source_csv, index_col=0)
        self.das_list = [int(d) for d in df.index[1:]]
        np_array = df.values
        df = df.transpose()

        for root_id, maximum_node_count, index_list in zip([int(l) for l in df.index[0:]], np_array[0], np_array[1:].T):
            single_root = SingleRoot(
                root_id = root_id, 
                das_list = self.das_list,
                maximum_node_count = maximum_node_count,
                index_list=index_list
            )
            self.RSA.append(single_root)

    def plot_all(self, fitting=False):
        return self.plot_each(self.RSA.get_root_id_list(), fitting=fitting)

    def plot_each(self, indexes:List[int], fitting=False, **kwargs):
        fig = plt.figure(figsize=(8,3))
        fig.subplots_adjust(bottom=0.2)
        cm = plt.get_cmap("Spectral")
        ax = fig.add_subplot(111)
        ax.set_xticks(np.arange(0, max(self.das_list)+1, 1))
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlim([-1, 28])
        ax.set_xlabel('days after sowing')
        ax.set_ylabel('relative potision from root tip')
        ax.invert_xaxis()
        
        for i, single_root in enumerate(self.RSA):
            if single_root.get_root_params()['elongation_period'] == 0:
                continue
            if single_root.root_id in indexes:
                additional_params = {'color': cm(i/(len(self.RSA)-1))}
                additional_params.update(kwargs)
                if fitting is False:
                    ax.plot(
                        single_root.das_list, 
                        single_root.relative_position_list, 
                        **additional_params
                    )
                else:
                    x_array, y_array = single_root.get_fitting_curve(das_list = list(range(0, max(self.das_list)+1)), y_clip=1)
                    ax.plot(
                        x_array, 
                        y_array,
                        **additional_params
                    )

        return fig

    def show_plot_each(self, indexes:List[int], fitting=False, **kwargs):
        fig = self.plot_each(indexes, fitting=fitting, **kwargs)
        plt.show()

    def save_plot_each(self, indexes:List[int], **kwargs):
        fig = self.plot_each(indexes, **kwargs)

        out_path = os.path.splitext(self.source_csv)[0]+'_each.png'
        fig.savefig(
            os.path.join(out_path),
            format="png", 
            dpi=300
        )

    def save_plot_all(self, fitting=False):
        fig = self.plot_all(fitting=fitting)

        if fitting==False:
            out_path = os.path.splitext(self.source_csv)[0]+'_raw.png'
        else:
            out_path = os.path.splitext(self.source_csv)[0]+'_fitted.png'

        fig.savefig(
            os.path.join(out_path),
            format="png", 
            dpi=300
        )

    def show_plot_all(self, fitting=False):
        _ = self.plot_all(fitting=fitting)
        plt.show()

@dataclass
class Quantification(object):
    rsa_vector: RSA_Vector
    fitting: Fitting

    @property
    def resolution(self):
        return self.rsa_vector.annotations.resolution()

    def get_df(self):
        params_list = []

        for base_node in self.rsa_vector:
            for root_node in base_node:
                polyline = np.array(root_node.interpolated_polyline())
                
                length = 0
                for co1, co2 in pairwise(polyline):
                    length += np.linalg.norm(co2-co1)

                length *= self.resolution/10

                root_id = root_node.ID
                params = {}
                for single_root in self.fitting.RSA:
                    if single_root.root_id == root_id:
                        params = single_root.get_root_params()
                        break

                if params['elongation_period']==0:
                    continue
                
                if len(params) != 0:
                    params.update({'length': length, 'elongation_rate': length/params['elongation_period']})
                
                params_list.append(params)

        df = pd.DataFrame(params_list)
        df.transpose()

        return df
