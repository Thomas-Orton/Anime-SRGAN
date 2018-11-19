from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import cv2
from keras.models import model_from_json
import keras.backend as K
import random
import cv2
from keras.utils import conv_utils
from keras.engine.topology import Layer
import time

small_rescale_dim=(360,480)

def create_high_res_video(file_in,file_out,video_blur=0.000001):
    model = load_model("generator")
    model_input_dim=model.layers[0].output_shape[1:3]
    model_output_dim=model.layers[-1].output_shape[1:3]
    transformation_function = get_video_processing_fn(model=model,
                                                                 model_input_dim=model_input_dim,
                                                                 model_output_dim=model_output_dim,
                                                                 blur_std=video_blur)
    process_video(file_in, file_out, transformation_function)





#util functions

'''
from https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981
'''

class PixelShuffler(Layer):
    def __init__(self, size=(2, 2), data_format='channels_last', **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs):

        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_last':
            batch_size, h, w, c = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, oh, ow, oc))
            return out

    def compute_output_shape(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_last':
            height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
            width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
            channels = input_shape[3]//(self.size[0] * self.size[1])

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            return (input_shape[0],
                    height,
                    width,
                    channels)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def rescale_image(image,image_dimension,interpolation=cv2.INTER_AREA):
    '''
        Rescale image
    '''
    return cv2.resize(image, image_dimension[::-1], interpolation=interpolation)#it seems that openCV uses dimensions WxH -- we'll use HxW, since that's consistent with the way videos list their resolution, and it's also consistent with np.shape

def reconstruct_image(model,X_downsample,model_input_dim,model_output_dim=None,patch_over_boundaries=True):
    patch_over_boundaries=True
    time_start=time.time()
    X_downsample=normalize_image(X_downsample)
    rescale_ratio=1
    if model_output_dim is not None:
        rescale_ratio=model_output_dim[0]//model_input_dim[0] ##must be integer
    Y=np.zeros(shape=(X_downsample.shape[0]*rescale_ratio,X_downsample.shape[1]*rescale_ratio,3))
    input_patches=[]
    patch_buffer_size_lr=16
    ycoord=0
    while ycoord<X_downsample.shape[0]:
        xcoord = 0
        ycoord-=patch_buffer_size_lr
        ycoord=max(0,ycoord)
        while xcoord<X_downsample.shape[1]:
            xcoord -= patch_buffer_size_lr
            xcoord = max(0, xcoord)
            #print(ycoord,xcoord)
            ycoord=min(ycoord,X_downsample.shape[0]-model_input_dim[0])
            xcoord = min(xcoord, X_downsample.shape[1] - model_input_dim[1])
            input_patches.append(X_downsample[ycoord:(ycoord+model_input_dim[0]),xcoord:(xcoord+model_input_dim[1]),:])
            xcoord+=model_input_dim[1]
        ycoord += model_input_dim[0]
    predict_array=input_patches
    #print(len(predict_array))
    time_after_extracting=time.time()
    predictions=np.array(model.predict(np.array(predict_array),batch_size=8))
    time_after_predictions=time.time()
    num_predictions=len(predict_array)

    ycoord = 0
    counter=0
    while ycoord < X_downsample.shape[0]*rescale_ratio:
        xcoord = 0
        ycoord -= patch_buffer_size_lr*rescale_ratio
        ycoord = max(0, ycoord)
        while xcoord < X_downsample.shape[1]*rescale_ratio:
            xcoord -= patch_buffer_size_lr*rescale_ratio
            xcoord = max(0, xcoord)

            ycoord = min(ycoord, X_downsample.shape[0]*rescale_ratio - model_input_dim[0]*rescale_ratio)
            xcoord = min(xcoord, X_downsample.shape[1]*rescale_ratio - model_input_dim[1]*rescale_ratio)
            #do cut and paste logic here
            # if we're on the boundaries, we should use the full image; otherwise, we use a slice of the image
            left_cut=(patch_buffer_size_lr//2)*rescale_ratio
            right_cut=(patch_buffer_size_lr//2)*rescale_ratio
            top_cut=(patch_buffer_size_lr//2)*rescale_ratio
            bottom_cut=(patch_buffer_size_lr//2)*rescale_ratio
            end_y_pred = -(patch_buffer_size_lr//2)*rescale_ratio
            end_x_pred = -(patch_buffer_size_lr//2)*rescale_ratio
            if patch_over_boundaries and ycoord==0:
                top_cut=0
            if patch_over_boundaries and ycoord==(X_downsample.shape[0] - model_input_dim[0])*rescale_ratio:
                bottom_cut=0
                end_y_pred = model_output_dim[0]*rescale_ratio
            if patch_over_boundaries and xcoord==0:
                left_cut=0
            if patch_over_boundaries and xcoord==(X_downsample.shape[1] - model_input_dim[1])*rescale_ratio:
                right_cut=0
                end_x_pred = model_output_dim[1]*rescale_ratio
            #print(ycoord,xcoord,top_cut,bottom_cut,left_cut,right_cut,predictions[counter].shape)
            Y[(ycoord + top_cut):(ycoord + model_input_dim[0] * rescale_ratio -bottom_cut),
            (xcoord + left_cut):(xcoord + model_input_dim[1] * rescale_ratio - right_cut), :] = predictions[counter, top_cut:end_y_pred,
                                                                                    left_cut:end_x_pred]

            #
            xcoord += model_input_dim[1]*rescale_ratio
            counter+=1
        ycoord += model_input_dim[0]*rescale_ratio

    time_after_inserting=time.time()
    return unnormalize_image(Y)

def get_video_processing_fn(model,model_input_dim,model_output_dim,blur_std):
    return lambda image : video_processing_upscale_frame_with_gan(image,model,model_input_dim,model_output_dim,blur_std)

def video_processing_upscale_frame_with_gan(image,model,model_input_dim,model_output_dim,blur_std):
    resized_image = rescale_image(image, small_rescale_dim, cv2.INTER_AREA)
    blurred_image=smooth_image(resized_image,kernel_size=3,blur_type="guass",std=blur_std)
    upscaled_image=reconstruct_image(model,blurred_image,model_input_dim,model_output_dim)
    upscaled_image=rescale_image(upscaled_image,(1080,1920),interpolation=cv2.INTER_AREA)
    return upscaled_image

def normalize_image(image):
    '''
    '''
    return image/127.5-1

def unnormalize_image(image,justScaleBy255=False):
    '''
    '''
    if justScaleBy255:
        image=image*255
    else:
        image=(image+1)*127.5;
    image=image.astype(np.uint8)
    image = np.clip(image, 0, 255)
    return image

def smooth_image(image,kernel_size=5,blur_type="guass",std=0):
    if kernel_size==1:
        return np.copy(image)
    if blur_type=="guass":
        ret =  cv2.GaussianBlur(image,(kernel_size,kernel_size),std)
    elif blur_type=="box":
        ret = cv2.blur(image, (kernel_size, kernel_size))
    return ret

def process_video(input_filename,output_filename,transformation):
    clip = VideoFileClip(input_filename)
    new_clip = clip.fl_image(transformation)
    new_clip.write_videofile(output_filename)


def load_model(model_name):
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json,custom_objects={'PixelShuffler':PixelShuffler()})
    # load weights into new model
    loaded_model.load_weights(model_name+".h5")
    return loaded_model

