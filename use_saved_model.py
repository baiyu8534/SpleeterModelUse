"""

直接sess.run  outputs_out = sess.run(outputs, feed_dict=self._get_input_provider().get_feed_dict(features, stft, audio_id))
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


def predict(waveform, audio_id):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "E:\\spleeter_model\\saved_model\\v1")
        graph = tf.get_default_graph()
        # feed_dict = {"audio_id:0": [feature.input_ids],
        #             "strided_slice_3:0": [feature.input_mask],
        #             "mix_stft:0": [feature.segment_ids]}
        # input_audio_id = sess.graph.get_tensor_by_name("audio_id:0")
        # input_strided_slice = sess.graph.get_tensor_by_name("strided_slice_3:0")
        input_mix_stft = sess.graph.get_tensor_by_name("mix_stft:0")

        output_accompaniment = sess.graph.get_tensor_by_name("mul_3:0")
        # output_audio_id = sess.graph.get_tensor_by_name("audio_id:0")
        output_vocals = sess.graph.get_tensor_by_name("mul_2:0")

        # print(input_audio_id)
        print(input_mix_stft)
        # print(input_strided_slice)

        print(output_accompaniment)
        # print(output_audio_id)
        print(output_vocals)

        # print(input_audio_id is output_audio_id)

        output_tensors = {
            "vocals": output_vocals,
            "accompaniment": output_accompaniment,
            # "audio_id": output_audio_id
        }

        stft = _stft(waveform)
        if stft.shape[-1] == 1:
            stft = np.concatenate([stft, stft], axis=-1)
        elif stft.shape[-1] > 2:
            stft = stft[:, :2]

        feed_dict = {
            # input_audio_id: audio_id,
            # input_strided_slice: audio_id,
            input_mix_stft: stft
        }

        """
        # alternative way
        feed_dict = {sess.graph.get_tensor_by_name("input_ids_1:0"): 
                              [feature.input_ids],
                    sess.graph.get_tensor_by_name("input_mask_1:0"):
                              [feature.input_mask],
                    sess.graph.get_tensor_by_name("segment_ids_1:0"):
                              [feature.segment_ids]}
        """
        outputs_out = sess.run(output_tensors, feed_dict=feed_dict)

        print(outputs_out)
        # 不行
        # with tf.Session() as sess:
        #     tflite_model = tf.lite.toco_convert(sess.graph_def, [input_mix_stft], [output_accompaniment,output_vocals])
        #     open("/Users/baiyu/Music/pretrained_models/transfrom_model/spleeter_v7.tflite", "wb").write(tflite_model)
        # save_model_to_freeze_2()


def transfrom_to_tflite():
    # 不行
    with tf.Session(graph=tf.Graph()) as sess:
        # saved_model_dir = '/Users/baiyu/Music/pretrained_models/transfrom_model/v7'
        model = tf.saved_model.loader.load(sess, ["serve"], "/Users/baiyu/Music/pretrained_models/transfrom_model/v7")
        print(dir(model))
        concrete_func = model.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        # print(model)
        # print(model)
        # concrete_func = model.signatures["tensorflow/serving/predict"]
        print(concrete_func)
        print(concrete_func.inputs[0])
        print(concrete_func.inputs[0])

        concrete_func.inputs[0].set_shape([1, 2049, 2])
        concrete_func.outputs[0].set_shape([1, 2049, 2])
        concrete_func.outputs[1].set_shape([1, 2049, 2])
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        # converter.experimental_new_converter = True
        tflite_model = converter.convert()
        open('/Users/baiyu/Music/pretrained_models/transfrom_model/spleeter_v7.tflite', 'wb').write(tflite_model)

def save_model_to_freeze():
    # 不行
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.saved_model import tag_constants

    input_saved_model_dir = "/Users/baiyu/Music/pretrained_models/transfrom_model/v7"
    output_node_names = "vocals,accompaniment"
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = tag_constants.SERVING
    output_graph_filename = '/Users/baiyu/Music/pretrained_models/transfrom_model/spleeter_v7_frozen_graph.pb'

    freeze_graph.freeze_graph(input_graph_filename,
                              input_saver_def_path,
                              input_binary,
                              checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_filename,
                              clear_devices,
                              "", "", "",
                              input_meta_graph,
                              input_saved_model_dir,
                              saved_model_tags)

def save_model_to_freeze_2():
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.saved_model import tag_constants

    saved_model_dir = "/Users/baiyu/Music/pretrained_models/transfrom_model/v7"
    output_graph_filename = "/Users/baiyu/Music/pretrained_models/transfrom_model/spleeter_v7_frozen_graph.pb"
    output_node_names = "mul_3:0,mul_2:0"
    # output_node_names = "vocals,accompaniment"
    initializer_nodes = ""

    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags=tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,

        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )

    print("SavedModel graph freezed!")

def main():
    music_path = "./chengdu.mp3"
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

    predict(waveform, music_path)
    # print(waveform)
    # print(len(waveform))
    # print(sample_rate)


if __name__ == "__main__":
    main()
    # transfrom_to_tflite()
    # save_model_to_freeze()
    # save_model_to_freeze_2()
    # import tensorflow_core
    # converter = tensorflow_core.lite.TFLiteConverter.from_saved_model("E:\\spleeter_model\\saved_model\\v1")
    # converter.experimental_new_converter = True
    # lite_model = converter.convert()
    #
    # # Format model file paths to store the file in the right directory:
    # lite_name ='spleeter.tflite'
    # out_path = "E:\\spleeter_model\\saved_model\\"  + lite_name
    #
    # # Write the converted model to disk:
    # open(out_path, "wb").write(lite_model)
    # 不是内部或外部命令，也不是可运行的程序
    # 或批处理文件。