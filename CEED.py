# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Address all TODOs and remove all explanatory comments
# Lint as: python3
"""CEED: California Earthquake Dataset for Machine Learning and Cloud Computing"""


from typing import Dict, List, Optional, Tuple, Union

import datasets
import fsspec
import h5py
import numpy as np
import torch


def _read_h5_attr(attrs, key, default=None):
    """读取 HDF5 属性；若 h5py 无法映射文件内浮点 dtype，__getitem__ 会抛错，此时返回 default。"""
    try:
        return attrs[key]
    except Exception:
        return default


def _as_float(x, default=np.nan):
    if x is None:
        return float(default)
    try:
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return float(default)


def _read_phase_attr_list(attrs, key: str):
    """读取台站 phase_* 属性并转为 list；整型/变长类型若无法映射到 NumPy，attrs[key] 会抛错。"""
    v = _read_h5_attr(attrs, key, None)
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        pass
    try:
        a = np.asarray(v)
        if a.ndim == 0:
            return [a.item()]
        return a.tolist()
    except Exception:
        return []


def _read_waveform_2d_via_h5d_read(ds, n0: int, n1: int) -> np.ndarray:
    """不经由 h5py 高级索引，直接用 H5Dread 读入 float32 缓冲区（由 HDF5 做类型转换）。"""
    from h5py import h5s

    dsid = ds.id
    out = np.ascontiguousarray(np.empty((n0, n1), dtype=np.float32))
    fspace = dsid.get_space()
    fspace.select_hyperslab((0, 0), (n0, n1))
    mspace = h5s.create_simple((n0, n1))
    dsid.read(mspace, fspace, out)
    return out


def _read_waveform_2d(ds, nt: int) -> np.ndarray:
    """读取台站波形 [C, T]。

    部分 H5 使用 h5py 无法映射到 NumPy 的 HDF5 浮点类型：``ds[...]`` / ``astype`` 会在
    构造 Reader 时失败。此时依次尝试 ``read_direct`` 与底层 ``DatasetID.read``，
    由 libhdf5 在读取时转换到内存中的 IEEE float32。
    """
    sh = ds.shape
    n0 = int(sh[0])
    n1 = min(int(nt), int(sh[1]))

    try:
        return np.asarray(ds[:, :nt], dtype=np.float32)
    except (ValueError, TypeError, OSError):
        pass

    out = np.empty((n0, n1), dtype=np.float32)
    try:
        ds.read_direct(out, np.s_[:, :n1])
        return out
    except (ValueError, TypeError, OSError):
        pass
    try:
        tmp = np.empty((n0, n1), dtype=np.float64)
        ds.read_direct(tmp, np.s_[:, :n1])
        return tmp.astype(np.float32, copy=False)
    except (ValueError, TypeError, OSError):
        pass

    if hasattr(ds, "astype"):
        for conv in (np.float32, np.float64):
            try:
                return np.asarray(ds.astype(conv)[:, :nt], dtype=np.float32)
            except (ValueError, TypeError, OSError):
                continue

    try:
        return _read_waveform_2d_via_h5d_read(ds, n0, n1)
    except Exception as e:
        raise RuntimeError(
            "无法将波形数据集转为 float32：文件内浮点类型与当前 h5py/NumPy 不兼容，"
            "且 read_direct / H5Dread 亦失败。可尝试升级 h5py 或检查 H5 是否损坏。"
        ) from e


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {CEED: California Earthquake Dataset for Machine Learning and Cloud Computing},
author={Zhu et al.},
year={2025}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_REPO_NC = "https://huggingface.co/datasets/AI4EPS/quakeflow_nc/resolve/main/waveform_h5"
_FILES_NC = [
""
]

_REPO_SC = "https://huggingface.co/datasets/AI4EPS/quakeflow_sc/resolve/main/waveform_h5"
_FILES_SC = []

# 仅包含2002年数据
_URLS_2002 = {
    "full": [f"{_REPO_NC}/{x}" for x in _FILES_NC],
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class CEED(datasets.GeneratorBasedBuilder):
    """CEED: A dataset of earthquake waveforms organized by earthquake events and based on the HDF5 format."""

    VERSION = datasets.Version("1.1.0")

    nt = 8192

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    # default config, you can change batch_size and num_stations_list when use `datasets.load_dataset`
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="station", version=VERSION, description="yield station-based samples one by one of whole dataset"
        ),
        datasets.BuilderConfig(
            name="event", version=VERSION, description="yield event-based samples one by one of whole dataset"
        ),
        datasets.BuilderConfig(
            name="station_train",
            version=VERSION,
            description="yield station-based samples one by one of training dataset",
        ),
        datasets.BuilderConfig(
            name="event_train", version=VERSION, description="yield event-based samples one by one of training dataset"
        ),
        datasets.BuilderConfig(
            name="station_test", version=VERSION, description="yield station-based samples one by one of test dataset"
        ),
        datasets.BuilderConfig(
            name="event_test", version=VERSION, description="yield event-based samples one by one of test dataset"
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        "station_test"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if (
            (self.config.name == "station")
            or (self.config.name == "station_train")
            or (self.config.name == "station_test")
        ):
            features = datasets.Features(
                {
                    "data": datasets.Array2D(shape=(3, self.nt), dtype="float32"),
                    "phase_time": datasets.Sequence(datasets.Value("string")),
                    "phase_index": datasets.Sequence(datasets.Value("int32")),
                    "phase_type": datasets.Sequence(datasets.Value("string")),
                    "phase_polarity": datasets.Sequence(datasets.Value("string")),
                    "begin_time": datasets.Value("string"),
                    "end_time": datasets.Value("string"),
                    "event_time": datasets.Value("string"),
                    "event_time_index": datasets.Value("int32"),
                    "event_location": datasets.Sequence(datasets.Value("float32")),
                    "station_location": datasets.Sequence(datasets.Value("float32")),
                },
            )
        elif (self.config.name == "event") or (self.config.name == "event_train") or (self.config.name == "event_test"):
            features = datasets.Features(
                {
                    "data": datasets.Array3D(shape=(None, 3, self.nt), dtype="float32"),
                    "phase_time": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "phase_index": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                    "phase_type": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "phase_polarity": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "begin_time": datasets.Value("string"),
                    "end_time": datasets.Value("string"),
                    "event_time": datasets.Value("string"),
                    "event_time_index": datasets.Value("int32"),
                    "event_location": datasets.Sequence(datasets.Value("float32")),
                    "station_location": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                },
            )
        else:
            raise ValueError(f"config.name = {self.config.name} is not in BUILDER_CONFIGS")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        
        # 支持本地目录：若传入 data_dir，则直接在该目录下查找 H5 文件
        data_dir = dl_manager.download_config.extract_dir if hasattr(dl_manager.download_config, "extract_dir") else None
        user_data_dir = getattr(self.config, "data_dir", None)
        local_dir = user_data_dir or data_dir

        if local_dir is not None:
            import os
            from glob import glob
            pattern = os.path.join(local_dir, "*.h5")
            files = sorted(glob(pattern))
            if not files:
                # 回退到远程下载
                urls = _URLS_2002["full"]
                files = dl_manager.download_and_extract(urls)
        else:
            # 仅下载2002年数据
            urls = _URLS_2002["full"]
            files = dl_manager.download_and_extract(urls)
        print(files)

        # 获取所有事件用于划分
        all_events = self._get_all_events(files)
        
        # 划分训练集和测试集 (80% 训练, 20% 测试)
        train_events, test_events = self._split_events(all_events, train_ratio=0.8)

        if self.config.name in ["station", "event"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": files,
                        "split": "train",
                        "selected_events": train_events,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": files, 
                        "split": "test",
                        "selected_events": test_events
                    },
                ),
            ]
        elif self.config.name in ["station_train", "event_train"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": files,
                        "split": "train",
                        "selected_events": train_events,
                    },
                ),
            ]
        elif self.config.name in ["station_test", "event_test"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": files,
                        "split": "test",
                        "selected_events": test_events
                    },
                ),
            ]
        else:
            raise ValueError("config.name is not in BUILDER_CONFIGS")

    def _get_all_events(self, files):
        """获取所有事件ID用于划分"""
        all_events = []
        for file in files:
            with fsspec.open(file, "rb") as fs:
                with h5py.File(fs, "r") as fp:
                    event_ids = list(fp.keys())
                    all_events.extend([(file, event_id) for event_id in event_ids])
        return all_events

    def _split_events(self, all_events, train_ratio=0.8):
        """划分事件为训练集和测试集"""
        import random
        random.seed(42)
        
        # 随机打乱事件
        shuffled_events = all_events.copy()
        random.shuffle(shuffled_events)
        
        # 按比例划分
        split_idx = int(len(shuffled_events) * train_ratio)
        train_events = shuffled_events[:split_idx]
        test_events = shuffled_events[split_idx:]
        
        print(f"总事件数: {len(all_events)}")
        print(f"训练集事件数: {len(train_events)}")
        print(f"测试集事件数: {len(test_events)}")
        
        return train_events, test_events

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split, selected_events=None):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # 创建文件到事件的映射
        file_events_map = {}
        for file, event_id in selected_events:
            if file not in file_events_map:
                file_events_map[file] = []
            file_events_map[file].append(event_id)

        for file in filepath:
            if file not in file_events_map:
                continue
                
            with fsspec.open(file, "rb") as fs:
                with h5py.File(fs, "r") as fp:
                    # 只处理选中的事件
                    event_ids = file_events_map[file]
                    for event_id in event_ids:
                        if event_id not in fp:
                            continue
                            
                        event = fp[event_id]
                        event_attrs = event.attrs
                        begin_time = _read_h5_attr(event_attrs, "begin_time", 0.0)
                        end_time = _read_h5_attr(event_attrs, "end_time", 0.0)
                        event_location = [
                            _as_float(_read_h5_attr(event_attrs, "longitude", np.nan)),
                            _as_float(_read_h5_attr(event_attrs, "latitude", np.nan)),
                            _as_float(_read_h5_attr(event_attrs, "depth_km", np.nan)),
                        ]
                        event_time = _read_h5_attr(event_attrs, "event_time", 0.0)
                        event_time_index = _read_h5_attr(event_attrs, "event_time_index", 0.0)
                        station_ids = list(event.keys())
                        if len(station_ids) == 0:
                            continue
                            
                        if ("station" in self.config.name):
                            waveforms = np.zeros([3, self.nt], dtype="float32")

                            for i, sta_id in enumerate(station_ids):
                                waveforms[:, : self.nt] = _read_waveform_2d(
                                    event[sta_id], self.nt
                                )
                                attrs = event[sta_id].attrs
                                phase_type = _read_phase_attr_list(attrs, "phase_type")
                                phase_time = _read_phase_attr_list(attrs, "phase_time")
                                phase_index = _read_phase_attr_list(attrs, "phase_index")
                                phase_polarity = _read_phase_attr_list(attrs, "phase_polarity")
                                _lon = _read_h5_attr(attrs, "longitude", np.nan)
                                _lat = _read_h5_attr(attrs, "latitude", np.nan)
                                _elev = _read_h5_attr(attrs, "elevation_m", 0.0)
                                station_location = [
                                    _as_float(_lon),
                                    _as_float(_lat),
                                    -_as_float(_elev, 0.0) / 1e3,
                                ]

                                yield f"{event_id}/{sta_id}", {
                                    "data": waveforms,
                                    "phase_time": phase_time,
                                    "phase_index": phase_index,
                                    "phase_type": phase_type,
                                    "phase_polarity": phase_polarity,
                                    "begin_time": begin_time,
                                    "end_time": end_time,
                                    "event_time": event_time,
                                    "event_time_index": event_time_index,
                                    "event_location": event_location,
                                    "station_location": station_location,
                                }

                        elif ("event" in self.config.name):
                            waveforms = np.zeros([len(station_ids), 3, self.nt], dtype="float32")
                            phase_type = []
                            phase_time = []
                            phase_index = []
                            phase_polarity = []
                            station_location = []

                            for i, sta_id in enumerate(station_ids):
                                waveforms[i, :, : self.nt] = _read_waveform_2d(
                                    event[sta_id], self.nt
                                )
                                attrs = event[sta_id].attrs
                                phase_type.append(_read_phase_attr_list(attrs, "phase_type"))
                                phase_time.append(_read_phase_attr_list(attrs, "phase_time"))
                                phase_index.append(_read_phase_attr_list(attrs, "phase_index"))
                                phase_polarity.append(_read_phase_attr_list(attrs, "phase_polarity"))
                                _lon = _read_h5_attr(attrs, "longitude", np.nan)
                                _lat = _read_h5_attr(attrs, "latitude", np.nan)
                                _elev = _read_h5_attr(attrs, "elevation_m", 0.0)
                                station_location.append(
                                    [
                                        _as_float(_lon),
                                        _as_float(_lat),
                                        -_as_float(_elev, 0.0) / 1e3,
                                    ]
                                )
                            yield event_id, {
                                "data": waveforms,
                                "phase_time": phase_time,
                                "phase_index": phase_index,
                                "phase_type": phase_type,
                                "phase_polarity": phase_polarity,
                                "begin_time": begin_time,
                                "end_time": end_time,
                                "event_time": event_time,
                                "event_time_index": event_time_index,
                                "event_location": event_location,
                                "station_location": station_location,
                            }