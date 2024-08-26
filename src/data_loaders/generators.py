import numpy as np
from tensorflow.keras.utils import Sequence
import itertools


class DataGenerator2D(Sequence):
    """
    For 2D model comparison
    """

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        # self.seq_length = seq_length

        self.indices = np.arange(self.x.shape[0])

        # Fixed window specifics
        self._h = 32
        self._ph1 = 6
        self._ph2 = self._ph1 + self._h

        self._w = 64
        self._pw1 = 2
        self._pw2 = self._pw1 + self._w

    def __len__(self):
        return int(np.floor(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Generate indexes of the batch
        current_indexes = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Crop the specified window!
        batch_x = self.x[current_indexes, self._ph1:self._ph2, self._pw1:self._pw2]
        batch_y = self.y[current_indexes, self._ph1:self._ph2, self._pw1:self._pw2]

        return batch_x, batch_y

    def on_epoch_end(self):
        # Shuffle indices after every epoch
        np.random.shuffle(self.indices)


class DataGenerator1(Sequence):
    """
    Basic predictors, fixed window
    """
    def __init__(self, x_set, y_set, batch_size, seq_length):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.start_indices = np.arange(self.x.shape[0] - self.seq_length)

        # Fixed window specifics
        self._h = 32
        self._ph1 = 6
        self._ph2 = self._ph1 + self._h

        self._w = 64
        self._pw1 = 2
        self._pw2 = self._pw1 + self._w

    def __len__(self):
        return int(np.floor(len(self.start_indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        low_i = idx * self.batch_size
        high_i = (idx + 1) * self.batch_size
        selected_start_inds = self.start_indices[low_i:high_i]

        sequence_indices = [np.arange(start_index, start_index+self.seq_length)
                            for start_index in selected_start_inds]
        batch_x = np.take(self.x, sequence_indices, axis=0)
        batch_y = np.take(self.y, sequence_indices, axis=0)

        # Crop the specified window!
        batch_x = batch_x[:, :, self._ph1:self._ph2, self._pw1:self._pw2]
        batch_y = batch_y[:, :, self._ph1:self._ph2, self._pw1:self._pw2]

        return batch_x, batch_y

    def on_epoch_end(self):
        # Shuffle indices after every epoch
        np.random.shuffle(self.start_indices)


class DataGenerator2(Sequence):
    """
    Basic predictors, moving window
    """
    def __init__(self, x_set, y_set, batch_size, seq_length, spatial_sample_multiplier):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.spatial_sample_multiplier = spatial_sample_multiplier

        self.start_indices = np.arange(self.x.shape[0] - self.seq_length)

        # Fixed window specifics
        self._h = 32
        self._w = 64

        area_h = 40
        area_w = 68
        self.h_shifts = np.arange(area_h - self._h + 1)  # Because :foo is exclusive
        self.w_shifts = np.arange(area_w - self._w + 1)  # Because :foo is exclusive

        self.spatial_combos = np.array(list(itertools.product(self.h_shifts, self.w_shifts)))

    def __len__(self):
        return int(np.floor(len(self.start_indices) / float(self.batch_size))) * self.spatial_sample_multiplier

    def __getitem__(self, idx):
        # Sample e.g. 2 times in a row the same days, to use them with different spatial window positions.
        temporal_idx = idx // self.spatial_sample_multiplier

        low_i = temporal_idx * self.batch_size
        high_i = (temporal_idx + 1) * self.batch_size
        selected_start_inds = self.start_indices[low_i:high_i]

        sequence_indices = [np.arange(start_index, start_index+self.seq_length)
                            for start_index in selected_start_inds]
        batch_x = np.take(self.x, sequence_indices, axis=0)
        batch_y = np.take(self.y, sequence_indices, axis=0)

        # Since we shuffle after each batch, we can simply take the first combo.
        hs = self.spatial_combos[0, 0]
        ws = self.spatial_combos[0, 1]
        # Crop the specified window!
        batch_x = batch_x[:, :, hs:self._h+hs, ws:self._w+ws]
        batch_y = batch_y[:, :, hs:self._h+hs, ws:self._w+ws]

        # Shuffle window shift indices after every batch
        np.random.shuffle(self.spatial_combos)

        return batch_x, batch_y

    def on_epoch_end(self):
        # Shuffle indices after every epoch
        np.random.shuffle(self.start_indices)


class DataGenerator3(Sequence):
    """
    Basic predictors plus elevation data, fixed window
    """
    def __init__(self, x_set, y_set, batch_size, seq_length, elevation_mat):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.elevation_mat = elevation_mat

        self.start_indices = np.arange(self.x.shape[0] - self.seq_length)

        # Fixed window specifics
        self._h = 32
        self._ph1 = 6
        self._ph2 = self._ph1 + self._h

        self._w = 64
        self._pw1 = 2
        self._pw2 = self._pw1 + self._w

    def __len__(self):
        return int(np.floor(len(self.start_indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        low_i = idx * self.batch_size
        high_i = (idx + 1) * self.batch_size
        selected_start_inds = self.start_indices[low_i:high_i]

        sequence_indices = [np.arange(start_index, start_index+self.seq_length)
                            for start_index in selected_start_inds]
        batch_x = np.take(self.x, sequence_indices, axis=0)
        batch_y = np.take(self.y, sequence_indices, axis=0)

        # Crop the specified window!
        batch_x = batch_x[:, :, self._ph1:self._ph2, self._pw1:self._pw2]
        batch_y = batch_y[:, :, self._ph1:self._ph2, self._pw1:self._pw2]

        # Reshape elevation data to a compatible size as batch_x
        elev = self.elevation_mat[self._ph1:self._ph2, self._pw1:self._pw2]
        # Create batch, sequence and channel dimension
        elev = np.expand_dims(elev, axis=(0, 1, -1))
        elev = np.repeat(np.repeat(elev, repeats=batch_x.shape[1], axis=1), repeats=batch_x.shape[0], axis=0)

        # Add elevation mat to last dimension at last position in batch_x only
        batch_x = np.concatenate((batch_x, elev), axis=-1)

        return batch_x, batch_y

    def on_epoch_end(self):
        # Shuffle indices after every epoch
        np.random.shuffle(self.start_indices)


class DataGenerator4(Sequence):
    """
    Basic predictors plus elevation data, moving window
    """
    def __init__(self, x_set, y_set, batch_size, seq_length, spatial_sample_multiplier, elevation_mat):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.spatial_sample_multiplier = spatial_sample_multiplier
        self.elevation_mat = elevation_mat

        self.start_indices = np.arange(self.x.shape[0] - self.seq_length)

        # Fixed window specifics
        self._h = 32
        self._w = 64

        area_h = 40
        area_w = 68
        self.h_shifts = np.arange(area_h - self._h + 1)  # Because :foo is exclusive
        self.w_shifts = np.arange(area_w - self._w + 1)  # Because :foo is exclusive

        self.spatial_combos = np.array(list(itertools.product(self.h_shifts, self.w_shifts)))

    def __len__(self):
        return int(np.floor(len(self.start_indices) / float(self.batch_size))) * self.spatial_sample_multiplier

    def __getitem__(self, idx):
        # Sample e.g. 2 times in a row the same days, to use them with different spatial window positions.
        temporal_idx = idx // self.spatial_sample_multiplier

        low_i = temporal_idx * self.batch_size
        high_i = (temporal_idx + 1) * self.batch_size
        selected_start_inds = self.start_indices[low_i:high_i]

        sequence_indices = [np.arange(start_index, start_index+self.seq_length)
                            for start_index in selected_start_inds]
        batch_x = np.take(self.x, sequence_indices, axis=0)
        batch_y = np.take(self.y, sequence_indices, axis=0)

        # Since we shuffle after each batch, we can simply take the first combo.
        hs = self.spatial_combos[0, 0]
        ws = self.spatial_combos[0, 1]
        # Crop the specified window!
        batch_x = batch_x[:, :, hs:self._h+hs, ws:self._w+ws]
        batch_y = batch_y[:, :, hs:self._h+hs, ws:self._w+ws]

        # Reshape elevation data to a compatible size as batch_x
        elev = self.elevation_mat[hs:self._h+hs, ws:self._w+ws]
        # Create batch, sequence and channel dimension
        elev = np.expand_dims(elev, axis=(0, 1, -1))
        elev = np.repeat(np.repeat(elev, repeats=batch_x.shape[1], axis=1), repeats=batch_x.shape[0], axis=0)

        # Add elevation mat to last dimension at last position in batch_x only
        batch_x = np.concatenate((batch_x, elev), axis=-1)

        # Shuffle window shift indices after every batch
        np.random.shuffle(self.spatial_combos)

        return batch_x, batch_y

    def on_epoch_end(self):
        # Shuffle indices after every epoch
        np.random.shuffle(self.start_indices)
