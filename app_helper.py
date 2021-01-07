import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import io
from PIL import Image, ImageOps

def vgg_layers(layer_names):
    
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    
    return model

def gram_matrix(input_tensor):

    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    return result/(num_locations)
    

class StyleContentModel(tf.keras.models.Model):
    
    def __init__(self, style_layers, content_layers):
        
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        
    def call(self, inputs):
        
        inputs = inputs *255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {content_name : value 
                        for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name : value 
                        for style_name, value in zip(self.style_layers, style_outputs)}
        
        return {'content' : content_dict,
                'style' : style_dict}
    
    
    
def load_img(path):
    """
     load an image and limit its maximum dimension to 512 pixels
    """
    max_dims = 512
    size = (512, 512)
    img = ImageOps.fit(path, size, Image.ANTIALIAS)
    img = np.asarray(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    img_shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    
    max_dim_img = max(img_shape)
    
    scale = max_dims/max_dim_img
    
    new_img_shape = tf.cast(img_shape*scale, tf.int32)
    
    img = tf.image.resize(img, new_img_shape)
    img = img[tf.newaxis,:]
    
    return img

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


extractor = StyleContentModel(style_layers, content_layers)


optimizer = tf.optimizers.Adam(learning_rate = 0.02, beta_1=0.99, epsilon=1e-1)


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)



num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def style_content_loss(outputs, targets, weights):
    style_targets = targets[0]
    content_targets = targets[1]
    
    content_weight = weights[0]
    style_weight = weights[1]
    
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = tf.add_n([tf.reduce_mean(style_outputs[name] - style_targets[name])**2 
                          for name in style_outputs.keys()])
    style_loss *= style_weight/num_style_layers
    
    content_loss = tf.add_n([tf.reduce_mean(content_outputs[name] - content_targets[name])**2 
                          for name in content_outputs.keys()])
    content_loss *= content_weight/num_content_layers
    
    loss = style_loss + content_loss
    return loss


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)



def train_step(image, targets, weights=(1e4, 1e-2)):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, targets, weights)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

    
    
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


    