import json
from os.path import join, dirname, split, exists

import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from skimage.io import imread

import sqlite3 as db

from skimage.transform import rescale


class DataRecord:

    def __init__(self, path: str, name: str, tags: list[str], description: str, documentation: str, authors: list[dict],
                 cite: list[dict], license: str = "BSD-2-Clause"):
        isinstance(name, str)
        self.name = name

        isinstance(tags, list)
        all(isinstance(t, str) for t in tags)
        self.tags = tags

        isinstance(description, str)
        self.description = description

        isinstance(documentation, str)
        self.documentation = documentation

        self.attachments = {}

        isinstance(authors, list)
        all(isinstance(a, dict) for a in authors)
        self.authors = authors

        isinstance(cite, list)
        all(isinstance(c, dict) for c in cite)
        self.cite = cite

        self.license = license

        self.path = path
        self.dir = join(self.path, name.replace(' ', '_') + ".zarr")

        self.loc = parse_url(self.dir)
        if exists(self.dir):
            self.zarr_container = zarr.open(self.dir, mode="w+")
        else:
            self.zarr_container = zarr.open(self.dir, mode="w")

        self.source_files_db = join(self.dir, "source_files.db")

    def serialize(self):
        rdf = {
            "format_version": "0.2.0",
            "name": self.name,
            "tags": self.tags,
            "description": self.description,
            "documentation": self.documentation,
            "attachments": self.attachments,
            "authors": self.authors,
            "cite": self.cite,
            "license": self.license,
            "type": "bias_net_comp.DataRecord",
            "version": "0.1.0"
        }
        with open(join(self.dir, self.name.replace(' ', '_') + '_rdf.json'), 'w') as f:
            json.dump(rdf, f, indent=4)

    @staticmethod
    def deserialize(json_name):
        with open(json_name, 'r') as f:
            rdf = json.load(f)

        return DefaultDataRecord(
            path=split(dirname(json_name))[0],
            name=rdf["name"],
            tags=rdf["tags"],
            description=rdf["description"],
            documentation=rdf["documentation"],
            authors=rdf["authors"],
            cite=rdf["cite"],
            license=rdf["license"]
        )


class DefaultDataRecord(DataRecord):

    def rescale_labeling(self, labeling, factors):
        scaled_labeling = rescale(np.zeros_like(labeling), scale=factors)

        for l in filter(None, np.unique(labeling)):
            mask = (labeling == l).astype(int)

            scaled_mask = rescale(mask, scale=factors, mode='reflect', anti_aliasing=True, preserve_range=True, order=1)

            scaled_labeling[scaled_mask > 0.5] = l

        return scaled_labeling

    def _create_dataset(self,
                        container,
                        name,
                        source_files,
                        target_files,
                        chunks,
                        axes,
                        data_spacing,
                        target_spacing,
                        min_shape,
                        source_dtype,
                        target_dtype,
                        target_name='nuclei'):
        assert source_dtype in [np.uint8, np.int8, np.int16, np.int32,
                                        np.int64, np.float64, np.float32, np.float16, np.complex64, np.complex128, bool]
        assert target_dtype in [np.uint8, np.int8, np.int16, np.int32,
                                        np.int64, np.float64, np.float32, np.float16, np.complex64, np.complex128, bool]

        if name in list(container.group_keys()):
            root = container[name]
        else:
            root = container.create_group(name)
            root.attrs['name'] = name
            root.attrs['type'] = 'ome-zarr'

        with db.connect(self.source_files_db) as conn:
            c = conn.cursor()
            c.execute(
                "CREATE TABLE IF NOT EXISTS {} ('source_file' text NOT NULL, 'target_file' text NOT NULL, 'index' INT NOT NULL)".format(
                    name)
            )

            index = len(root)
            if np.all(target_spacing == data_spacing):
                factors = np.array((1,) * len(target_spacing))
            else:
                factors = np.array(target_spacing) / np.array(data_spacing)

            for i, (src_f, trg_f) in enumerate(zip(source_files, target_files)):

                assert exists(src_f), "File {} could not be found.".format(src_f)
                assert exists(trg_f), "File {} could not be found.".format(trg_f)

                src_file_exists = DefaultDataRecord.file_in_db(c, src_f)
                trg_file_exists = DefaultDataRecord.file_in_db(c, trg_f)

                if not src_file_exists and not trg_file_exists:
                    grp = root.create_group(str(index))
                    labels_grp = grp.create_group('labels')
                    label_grp = labels_grp.create_group(target_name)
                    self.add_image(axes, chunks, data_spacing, source_dtype, factors, grp, min_shape, src_f,
                                   DefaultDataRecord.mi_ma_fun)
                    if target_dtype in [np.uint8, np.int8, np.int16, np.int32,
                                        np.int64]:
                        self.add_label_image(axes, chunks, data_spacing, target_dtype, factors, label_grp, min_shape,
                                             trg_f)
                    else:
                        self.add_image(axes, chunks, data_spacing, target_dtype, factors, label_grp, min_shape, trg_f,
                                       DefaultDataRecord.mi_ma_fun)

                    c.execute("INSERT INTO {} (source_file, target_file, [index]) VALUES ('{}', '{}', {})".format(name, src_f, trg_f, index))
                    index += 1

            self.attachments[name] = join(".", name)

    def add_image(self, axes, chunks, data_spacing, dtype, factors, grp, min_shape, file, mi_ma_fun):
        img = imread(file).astype(dtype)

        img = DefaultDataRecord.normalize_min_max(img, *mi_ma_fun(img))

        if np.any(factors != 1):
            img = rescale(img, scale=factors, mode='reflect', anti_aliasing=True, preserve_range=True,
                          order=1).astype(img.dtype)

        img = self.pad(img, min_shape, mode='reflect')

        if "c" not in axes:
            img = img[np.newaxis]
            axes = ("c", ) + axes
            chunks = (1, ) + chunks
            data_spacing = (1, ) + data_spacing
            factors = np.concatenate([np.array([1]), factors])

        write_image(image=img, group=grp, chunks=chunks, axes=axes, scaler=Scaler(downscale=1, max_layer=0))
        grp.attrs['dataspecs'] = {
            "source_file": file,
            "shape": img.shape,
            "axes": axes,
            "data_spacing": tuple(np.array(data_spacing, np.float32) * factors)
        }

    def pad(self, img, min_shape, mode='reflect', constant_values=-1):
        if np.any(img.shape < min_shape):
            pad_width = []
            for img_s, min_s in zip(img.shape, min_shape):
                margin = min_s - img_s
                pad_width.append([margin // 2, min_s - margin // 2])
            img = np.pad(img, pad_width=pad_width, mode=mode, constant_values=constant_values)
        return img

    def add_label_image(self, axes, chunks, data_spacing, dtype, factors, grp, min_shape, file):
        img = imread(file).astype(dtype)
        if np.any(factors != 1):
            img = self.rescale_labeling(img, factors=factors).astype(dtype)

        img = self.pad(img, min_shape, mode='constant', constant_values=-1)

        if "c" not in axes:
            img = img[np.newaxis]
            axes = ("c", ) + axes
            chunks = (1, ) + chunks
            data_spacing = (1, ) + data_spacing
            factors = np.concatenate([np.array([1]), factors])

        write_image(image=img, group=grp, chunks=chunks, axes=axes, scaler=Scaler(downscale=1, max_layer=0))
        grp.attrs['dataspecs'] = {
            "source_file": file,
            "shape": img.shape,
            "axes": axes,
            "data_spacing": tuple(np.array(data_spacing, dtype=np.float32) * factors)
        }

    @staticmethod
    def mi_ma_fun(img):
        mi, ma = np.quantile(img, [0.03, 0.998])
        return mi, ma

    @staticmethod
    def normalize_min_max(data, mi, ma):
        return (data - mi) / (ma - mi)

    @staticmethod
    def file_in_db(c, file):
        tables = [t[0] for t in c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]

        for table in tables:
            if DefaultDataRecord.source_file_exists(c, table, file) or DefaultDataRecord.target_file_exists(c, table,
                                                                                                            file):
                return True

        return False

    @staticmethod
    def source_file_exists(c, table, file):
        return bool(c.execute(
            "SELECT COUNT(1) FROM '{}' WHERE source_file = '{}'".format(table, file)
        ).fetchall()[0][0])

    @staticmethod
    def target_file_exists(c, table, file):
        return bool(c.execute(
            "SELECT COUNT(1) FROM '{}' WHERE target_file = '{}'".format(table, file)
        ).fetchall()[0][0])

    def create_train_val_test_dataset(self, source_files, target_files, chunks, axes, data_spacing, target_spacing,
                                      min_shape,
                                      val_fraction,
                                      test_fration,
                                      source_dtype=np.float32, target_dtype=np.uint16,
                                      seed=42):
        assert len(source_files) == len(target_files)
        assert len(source_files) > 3

        np.random.seed(seed)
        random_idx = np.random.permutation(len(source_files))
        n_val_samples = max(1, int(val_fraction * len(source_files)))
        n_test_samples = max(1, int(test_fration * len(source_files)))
        n_train_samples = len(source_files) - n_val_samples - n_test_samples

        source_files = np.array(source_files)
        target_files = np.array(target_files)

        self.add_train_data(source_files=source_files[random_idx[:n_train_samples]],
                            target_files=target_files[random_idx[:n_train_samples]],
                            chunks=chunks,
                            axes=axes,
                            data_spacing=data_spacing,
                            target_spacing=target_spacing,
                            min_shape=min_shape,
                            source_dtype=source_dtype,
                            target_dtype=target_dtype)

        self.add_val_data(source_files=source_files[random_idx[n_train_samples:n_train_samples + n_val_samples]],
                          target_files=target_files[random_idx[n_train_samples:n_train_samples + n_val_samples]],
                          chunks=chunks,
                          axes=axes,
                          data_spacing=data_spacing,
                          target_spacing=target_spacing,
                          min_shape=min_shape,
                          source_dtype=source_dtype,
                          target_dtype=target_dtype)

        self.add_test_data(source_files=source_files[random_idx[-n_test_samples:]],
                           target_files=target_files[random_idx[-n_test_samples:]],
                           chunks=chunks,
                           axes=axes,
                           data_spacing=data_spacing,
                           target_spacing=target_spacing,
                           min_shape=min_shape,
                           source_dtype=source_dtype,
                           target_dtype=target_dtype)

    def add_train_data(self, source_files, target_files, chunks, axes, data_spacing, target_spacing, min_shape,
                       source_dtype=np.float32, target_dtype=np.uint16):
        assert len(source_files) == len(target_files)
        self._create_dataset(self.zarr_container, name="train_data",
                             source_files=source_files,
                             target_files=target_files,
                             chunks=chunks,
                             axes=axes,
                             data_spacing=data_spacing,
                             target_spacing=target_spacing,
                             min_shape=min_shape,
                             source_dtype=source_dtype,
                             target_dtype=target_dtype)

    def add_val_data(self, source_files, target_files, chunks, axes, data_spacing, target_spacing, min_shape,
                     source_dtype=np.float32, target_dtype=np.uint16):
        assert len(source_files) == len(target_files)
        self._create_dataset(self.zarr_container, name="val_data",
                             source_files=source_files,
                             target_files=target_files,
                             chunks=chunks,
                             axes=axes,
                             data_spacing=data_spacing,
                             target_spacing=target_spacing,
                             min_shape=min_shape,
                             source_dtype=source_dtype,
                             target_dtype=target_dtype)

    def add_test_data(self, source_files, target_files, chunks, axes, data_spacing, target_spacing, min_shape,
                      source_dtype=np.float32, target_dtype=np.uint16):
        assert len(source_files) == len(target_files)
        self._create_dataset(self.zarr_container, name="test_data",
                             source_files=source_files,
                             target_files=target_files,
                             chunks=chunks,
                             axes=axes,
                             data_spacing=data_spacing,
                             target_spacing=target_spacing,
                             min_shape=min_shape,
                             source_dtype=source_dtype,
                             target_dtype=target_dtype)
