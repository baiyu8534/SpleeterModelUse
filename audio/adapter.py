#!/usr/bin/env python
# coding: utf8

""" AudioAdapter class defintion. """

import subprocess

from abc import ABC, abstractmethod

# pylint: disable=import-error
import numpy as np
import tensorflow as tf

# from tensorflow.contrib.signal import stft, hann_window
# pylint: enable=import-error

# from .. import SpleeterError
# from ..utils.logging import get_logger

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


class AudioAdapter(ABC):
    """ An abstract class for manipulating audio signal. """

    # Default audio adapter singleton instance.
    DEFAULT = None

    @abstractmethod
    def load(
            self, audio_descriptor, offset, duration,
            sample_rate, dtype=np.float32):
        """ Loads the audio file denoted by the given audio descriptor
        and returns it data as a waveform. Aims to be implemented
        by client.

        :param audio_descriptor:    Describe song to load, in case of file
                                    based audio adapter, such descriptor would
                                    be a file path.
        :param offset:              Start offset to load from in seconds.
        :param duration:            Duration to load in seconds.
        :param sample_rate:         Sample rate to load audio with.
        :param dtype:               Numpy data type to use, default to float32.
        :returns:                   Loaded data as (wf, sample_rate) tuple.
        """
        pass

    def load_tf_waveform(
            self, audio_descriptor,
            offset=0.0, duration=1800., sample_rate=44100,
            dtype=b'float32', waveform_name='waveform'):
        """ Load the audio and convert it to a tensorflow waveform.

        :param audio_descriptor:    Describe song to load, in case of file
                                    based audio adapter, such descriptor would
                                    be a file path.
        :param offset:              Start offset to load from in seconds.
        :param duration:            Duration to load in seconds.
        :param sample_rate:         Sample rate to load audio with.
        :param dtype:               Numpy data type to use, default to float32.
        :param waveform_name:       (Optional) Name of the key in output dict.
        :returns:                   TF output dict with waveform as
                                    (T x chan numpy array)  and a boolean that
                                    tells whether there were an error while
                                    trying to load the waveform.
        """
        # Cast parameters to TF format.
        offset = tf.cast(offset, tf.float64)
        duration = tf.cast(duration, tf.float64)

        # Defined safe loading function.
        def safe_load(path, offset, duration, sample_rate, dtype):
            # logger = get_logger()
            # logger.info(
            #     f'Loading audio {path} from {offset} to {offset + duration}')
            print(f'Loading audio {path} from {offset} to {offset + duration}')
            try:
                (data, _) = self.load(
                    path.numpy(),
                    offset.numpy(),
                    duration.numpy(),
                    sample_rate.numpy(),
                    dtype=dtype.numpy())
                # logger.info('Audio data loaded successfully')
                print('Audio data loaded successfully')
                return (data, False)
            except Exception as e:
                # logger.exception(
                #     'An error occurs while loading audio',
                #     exc_info=e)
                print('An error occurs while loading audio')
            return (np.float32(-1.0), True)

        # Execute function and format results.
        results = tf.py_function(
            safe_load,
            [audio_descriptor, offset, duration, sample_rate, dtype],
            (tf.float32, tf.Variable)),
        waveform, error = results[0]
        return {
            waveform_name: waveform,
            f'{waveform_name}_error': error
        }

    @abstractmethod
    def save(
            self, path, data, sample_rate,
            codec=None, bitrate=None):
        """ Save the given audio data to the file denoted by
        the given path.

        :param path: Path of the audio file to save data in.
        :param data: Waveform data to write.
        :param sample_rate: Sample rate to write file in.
        :param codec: (Optional) Writing codec to use.
        :param bitrate: (Optional) Bitrate of the written audio file.
        """
        pass


def get_default_audio_adapter():
    """ Builds and returns a default audio adapter instance.

    :returns: An audio adapter instance.
    """
    if AudioAdapter.DEFAULT is None:
        from .ffmpeg import FFMPEGProcessAudioAdapter
        AudioAdapter.DEFAULT = FFMPEGProcessAudioAdapter()
    return AudioAdapter.DEFAULT


