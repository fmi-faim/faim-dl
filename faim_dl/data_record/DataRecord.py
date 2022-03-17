import json
from os.path import join, splitext, basename, dirname, split, exists

import numpy as np
import pandas as pd
import zarr
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from skimage.io import imread
import numcodecs

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
                        dtype,
                        compressor,
                        train,
                        val,
                        target_name='nuclei'):
        if name in list(container.group_keys()):
            root = container[name]
        else:
            root = container.create_group(name)
            root.attrs['name'] = name
            root.attrs['type'] = 'ome-zarr'

        offset = len(root)
        if np.all(target_spacing == data_spacing):
            factors = (1,) * len(target_spacing)
        else:
            factors = target_spacing / data_spacing

        for i, (src_f, trg_f) in enumerate(zip(source_files, target_files)):
            grp = root.create_group(str(offset + i))
            labels_grp = grp.create_group('labels')
            label_grp = labels_grp.create_group(target_name)

            src = imread(src_f).astype(dtype)
            trg = imread(trg_f).astype(dtype)

            if np.any(factors != 1):
                src = rescale(src, scale=factors, mode='reflect', anti_aliasing=True, preserve_range=True,
                              order=1).astype(src.dtype)
                trg = self.rescale_labeling(trg, factors=factors).astype(trg.dtype)

            if np.any(src.shape < min_shape):
                src = np.pad()

            write_image(image=src, group=grp, chunks=chunks, axes=axes, scaler=Scaler(downscale=1, max_layer=0))
            write_image(image=trg, group=label_grp, chunks=chunks, axes=axes,
                        scaler=Scaler(downscale=1, max_layer=0))

            grp.attrs['dataspecs'] = {
                "source_file": src_f,
                "target_file": trg_f,
                "shape": src.shape,
                "axes": axes,
                "data_spacing": data_spacing
            }

        self.attachments[name] = join(".", name)

        return pd.DataFrame({"source_file": source_files, "target_file": target_files})

    def create_train_val_test_dataset(self, source_files, target_files, chunks, axes, data_spacing, target_spacing,
                                      min_shape,
                                      val_fraction,
                                      test_fration,
                                      dtype=np.uint16, compressor=numcodecs.Blosc(), seed=42):
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
                            dtype=dtype,
                            compressor=compressor)

        self.add_val_data(source_files=source_files[random_idx[n_train_samples:n_train_samples + n_val_samples]],
                          target_files=target_files[random_idx[n_train_samples:n_train_samples + n_val_samples]],
                          chunks=chunks,
                          axes=axes,
                          data_spacing=data_spacing,
                          target_spacing=target_spacing,
                          min_shape=min_shape,
                          dtype=dtype,
                          compressor=compressor)

        self.add_test_data(source_files=source_files[random_idx[-n_test_samples:]],
                           target_files=target_files[random_idx[-n_test_samples:]],
                           chunks=chunks,
                           axes=axes,
                           data_spacing=data_spacing,
                           target_spacing=target_spacing,
                           min_shape=min_shape,
                           dtype=dtype,
                           compressor=compressor)

    def save2db(self, files, table_name):
        with db.connect(self.source_files_db) as conn:
            c = conn.cursor()
            c.execute(
                "CREATE TABLE IF NOT EXISTS {} ('source_file' text NOT NULL, 'target_file' text NOT NULL)".format(
                    table_name)
            )
            files.to_sql(table_name, conn, if_exists="append", index=False)

    def add_train_data(self, source_files, target_files, chunks, axes, data_spacing, target_spacing, min_shape,
                       dtype=np.uint16, compressor=numcodecs.Blosc()):
        assert len(source_files) == len(target_files)
        files = self._create_dataset(self.zarr_container, name="train_data",
                                     source_files=source_files,
                                     target_files=target_files,
                                     chunks=chunks,
                                     axes=axes,
                                     data_spacing=data_spacing,
                                     target_spacing=target_spacing,
                                     min_shape=min_shape,
                                     dtype=dtype,
                                     compressor=compressor,
                                     train=True, val=False)

        self.save2db(files, table_name="train_files")

    def add_val_data(self, source_files, target_files, chunks, axes, data_spacing, target_spacing, min_shape,
                     dtype=np.uint16, compressor=numcodecs.Blosc()):
        assert len(source_files) == len(target_files)
        files = self._create_dataset(self.zarr_container, name="val_data",
                                     source_files=source_files,
                                     target_files=target_files,
                                     chunks=chunks,
                                     axes=axes,
                                     data_spacing=data_spacing,
                                     target_spacing=target_spacing,
                                     min_shape=min_shape,
                                     dtype=dtype,
                                     compressor=compressor,
                                     train=False, val=True)

        self.save2db(files, table_name="val_files")

    def add_test_data(self, source_files, target_files, chunks, axes, data_spacing, target_spacing, min_shape,
                      dtype=np.uint16, compressor=numcodecs.Blosc()):
        assert len(source_files) == len(target_files)
        files = self._create_dataset(self.zarr_container, name="test_data",
                                     source_files=source_files,
                                     target_files=target_files,
                                     chunks=chunks,
                                     axes=axes,
                                     data_spacing=data_spacing,
                                     target_spacing=target_spacing,
                                     min_shape=min_shape,
                                     dtype=dtype,
                                     compressor=compressor,
                                     train=False, val=False)

        self.save2db(files, table_name="test_files")
