"""

直接sess.run  outputs_out = sess.run(outputs, feed_dict=_get_input_provider().get_feed_dict(features, stft, audio_id))
feed_dict: {'mix_stft': <tf.Tensor 'mix_stft:0' shape=(?, 2049, 2) dtype=complex64>, 'audio_id': <tf.Tensor 'audio_id:0' shape=<unknown> dtype=string>, 'mix_spectrogram': <tf.Tensor 'strided_slice_3:0' shape=(?, 512, 1024, 2) dtype=float32>}
outputs : {'vocals': <tf.Tensor 'mul_2:0' shape=(?, 2049, 2) dtype=complex64>, 'accompaniment': <tf.Tensor 'mul_3:0' shape=(?, 2049, 2) dtype=complex64>, 'audio_id': <tf.Tensor 'audio_id:0' shape=<unknown> dtype=string>}

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['audio_id'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: audio_id:0
    inputs['mix_spectrogram'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 512, 1024, 2)
        name: strided_slice_3:0
    inputs['mix_stft'] tensor_info:
        dtype: DT_COMPLEX64
        shape: (-1, 2049, 2)
        name: mix_stft:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['accompaniment'] tensor_info:
        dtype: DT_COMPLEX64
        shape: (-1, 2049, 2)
        name: mul_3:0
    outputs['audio_id'] tensor_info:
        dtype: DT_STRING
        shape: unknown_rank
        name: audio_id:0
    outputs['vocals'] tensor_info:
        dtype: DT_COMPLEX64
        shape: (-1, 2049, 2)
        name: mul_2:0
  Method name is: tensorflow/serving/predict



直接按spleeter中构造数据的方式构造feed_dict

读取保存的savedmodel模型

构造feed_dict

获取输出数据
"""

import tensorflow as tf
# from .audio.ffmpeg import FFMPEGProcessAudioAdapter
import audio.ffmpeg as ffmpeg
import numpy as np
from librosa.core import stft, istft
from scipy.signal.windows import hann

import os

from multiprocessing import Pool




def get_wavefrom_and_audioid():
    pass


def _stft(data, inverse=False, length=None):
    """
    Single entrypoint for both stft and istft. This computes stft and istft with librosa on stereo data. The two
    channels are processed separately and are concatenated together in the result. The expected input formats are:
    (n_samples, 2) for stft and (T, F, 2) for istft.
    :param data: np.array with either the waveform or the complex spectrogram depending on the parameter inverse
    :param inverse: should a stft or an istft be computed.
    :return: Stereo data as numpy array for the transform. The channels are stored in the last dimension
    """
    assert not (inverse and length is None)
    data = np.asfortranarray(data)
    N = 4096
    H = 1024
    win = hann(N, sym=False)
    fstft = istft if inverse else stft
    win_len_arg = {"win_length": None, "length": length} if inverse else {"n_fft": N}
    n_channels = data.shape[-1]
    out = []
    for c in range(n_channels):
        d = data[:, :, c].T if inverse else data[:, c]
        s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
        s = np.expand_dims(s.T, 2 - inverse)
        out.append(s)
    if len(out) == 1:
        return out[0]
    return np.concatenate(out, axis=2 - inverse)


def save_to_file(
        sources,
        codec='wav', audio_adapter=ffmpeg.FFMPEGProcessAudioAdapter(),
        bitrate='128k', synchronous=True):
    """ export dictionary of sources to files.

    :param sources:             Dictionary of sources to be exported. The
                                keys are the name of the instruments, and
                                the values are Nx2 numpy arrays containing
                                the corresponding intrument waveform, as
                                returned by the separate method
    :param audio_descriptor:    Describe song to separate, used by audio
                                adapter to retrieve and load audio data,
                                in case of file based audio adapter, such
                                descriptor would be a file path.
    :param destination:         Target directory to write output to.
    :param filename_format:     (Optional) Filename format.
    :param codec:               (Optional) Export codec.
    :param audio_adapter:       (Optional) Audio adapter to use for I/O.
    :param bitrate:             (Optional) Export bitrate.
    :param synchronous:         (Optional) True is should by synchronous.

    """

    # filename = "chengdu.mp3"
    pool = Pool()
    tasks = []
    for instrument, data in sources.items():
        path = "./out/"+instrument + "." + codec

        if pool:
            task = pool.apply_async(audio_adapter.save, (
                path,
                data,
                44100,
                codec,
                bitrate))
            tasks.append(task)
        else:
            audio_adapter.save(path, data, 44100, codec, bitrate)
    if synchronous and pool:
        while len(tasks) > 0:
            task = tasks.pop()
            task.get()
            task.wait(timeout=200)


def predict(waveform, audio_id):
    with tf.Session(graph=tf.Graph()) as sess:

        # with open('E:\\spleeter_freeze_v1.pb', 'rb') as f:
        with open('E:\\freeze_v1.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # tf.saved_model.loader.load(sess, ["serve"], "E:\\spleeter_model\\saved_model\\v1")
        # graph = tf.get_default_graph()
        # feed_dict = {"audio_id:0": [feature.input_ids],
        #             "strided_slice_3:0": [feature.input_mask],
        #             "mix_stft:0": [feature.segment_ids]}
        # input_audio_id = sess.graph.get_tensor_by_name("audio_id:0")
        # input_strided_slice = sess.graph.get_tensor_by_name("strided_slice_3:0")
        # input_mix_stft = sess.graph.get_tensor_by_name("mix_stft:0")
        #
        # output_accompaniment = sess.graph.get_tensor_by_name("mul_3:0")
        # # output_audio_id = sess.graph.get_tensor_by_name("audio_id:0")
        # output_vocals = sess.graph.get_tensor_by_name("mul_2:0")
        #
        # # print(input_audio_id)
        # print(input_mix_stft)
        # # print(input_strided_slice)
        #
        # print(output_accompaniment)
        # # print(output_audio_id)
        # print(output_vocals)
        #
        # # print(input_audio_id is output_audio_id)
        #
        # output_tensors = {
        #     "vocals": output_vocals,
        #     "accompaniment": output_accompaniment,
        #     # "audio_id": output_audio_id
        # }

        stft = _stft(waveform)
        if stft.shape[-1] == 1:
            stft = np.concatenate([stft, stft], axis=-1)
        elif stft.shape[-1] > 2:
            stft = stft[:, :2]

        # print(stft)

        # feed_dict = {
        #     # input_audio_id: audio_id,
        #     # input_strided_slice: audio_id,
        #     input_mix_stft: stft
        # }

        """
        # alternative way
        feed_dict = {sess.graph.get_tensor_by_name("input_ids_1:0"): 
                              [feature.input_ids],
                    sess.graph.get_tensor_by_name("input_mask_1:0"):
                              [feature.input_mask],
                    sess.graph.get_tensor_by_name("segment_ids_1:0"):
                              [feature.segment_ids]}
        """
        outputs_out = tf.import_graph_def(graph_def,
                                          input_map={'mix_stft:0': stft},
                                          return_elements=['mul_2:0', 'mul_3:0'])
        out = sess.run(outputs_out)
        # outputs_out = sess.run(output_tensors, feed_dict=feed_dict)

        # print(outputs_out)
        #
        # print(out)

        out_dict = dict(zip(["vocals", "accompaniment"], out))
        outdata = {}
        outdata["vocals"] = _stft(out_dict["vocals"], inverse=True, length=waveform.shape[0])
        outdata["accompaniment"] = _stft(out_dict["accompaniment"], inverse=True, length=waveform.shape[0])
        save_to_file(outdata)


def main():
    music_path = "./natural.mp3"
    output_path = "./output_music"

    audio_adapter = ffmpeg.FFMPEGProcessAudioAdapter()
    offset = 0
    duration = 600.
    codec = 'wav'
    bitrate = '128k'
    filename_format = '{filename}/{instrument}.{codec}'
    synchronous = False

    waveform, sample_rate = audio_adapter.load(
        music_path,
        offset=offset,
        duration=duration,
        sample_rate=44100)

    # print(waveform)

    predict(waveform, music_path)
    print(waveform)
    print(len(waveform))
    print(sample_rate)


if __name__ == "__main__":
    import time
    time_start = time.clock()
    main()
    # 3.0126122e-03 + 0.0000000e+00j, 2.8952009e-03 + 0
    time_end = time.clock()
    print("运行时间：",time_start-time_end)