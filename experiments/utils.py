from tkinter.ttk import Progressbar
import cv2
import lime
import numpy as np
from skimage.color import rgb2gray
from tqdm.auto import tqdm
import copy
from lime.lime_image import LimeImageExplainer
import tensorflow as tf

def resize_mask(mask, new_size):
  return cv2.resize(mask, dsize=new_size, interpolation=cv2.INTER_AREA)

def discretize_mask(mask):
  return (mask > 0.1).astype(int)


def mask_image(base_img, replacement_img, mask):
  inverse_mask = mask * (-1) + 1
  base = base_img * inverse_mask
  replacement = replacement_img * mask
  return base + replacement

def lime_ds(base_ds, replacement_ds, model):
    explainer = GrayLimeImageExplainer()
    xai_imgs_list = []
    for idx, img in enumerate(tqdm(base_ds)):
        explanation = explainer.explain_instance(img.astype('double'), model.predict, hide_color=0, num_samples=10)
        _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=False)
        if np.any(mask):
            xai_img = mask_image(img, replacement_ds[idx], mask)
        else:
            xai_img = replacement_ds[idx]
        xai_imgs_list.append(xai_img)
    return np.concatenate(xai_imgs_list).reshape(10000, 28, 28)

class GrayLimeImageExplainer(LimeImageExplainer):
  def data_labels(self, image, fudged_image, segments, classifier_fn, num_samples, batch_size=10):
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            temp = rgb2gray(temp) #TODO: my modification
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def gradcam_ds(base_ds, replacement_ds, model, last_conv):
    xai_img_list = []
    for idx, img in enumerate(tqdm(base_ds)):
        heatmap = make_gradcam_heatmap(img.reshape(1,28,28), model, last_conv)
        mask = discretize_mask(resize_mask(heatmap, (28,28)))
        xai_img = mask_image(img, replacement_ds[idx], mask)
        xai_img_list.append(xai_img)
    return np.concatenate(xai_img_list).reshape(10000, 28, 28) 

def random_mask(base_ds, replacement_ds):
    xai_img_list = []
    for idx, img in enumerate(tqdm(base_ds)):
        mask = (np.random.rand(28,28) > 0.5).astype(int)
        xai_img = mask_image(img, replacement_ds[idx], mask)
        xai_img_list.append(xai_img)
    return np.concatenate(xai_img_list).reshape(10000, 28, 28)

def center_mask(base_ds, replacement_ds):
    xai_img_list = []
    mask = np.array([1 if 4 < x < 24 and 4 < y < 24 else 0 for x in range(28) for y in range(28)]).reshape(28,28)
    for idx, img in enumerate(tqdm(base_ds)):
        xai_img = mask_image(img, replacement_ds[idx], mask)
        xai_img_list.append(xai_img)
    return np.concatenate(xai_img_list).reshape(10000, 28, 28)

def margin_mask(base_ds, replacement_ds):
    xai_img_list = []
    mask = np.array([0 if 4 < x < 24 and 4 < y < 24 else 1 for x in range(28) for y in range(28)]).reshape(28,28)
    for idx, img in enumerate(tqdm(base_ds)):
        xai_img = mask_image(img, replacement_ds[idx], mask)
        xai_img_list.append(xai_img)
    return np.concatenate(xai_img_list).reshape(10000, 28, 28)
    
def get_masks(img, model, last_conv = 'conv2d_3'):
    explainer = GrayLimeImageExplainer()
    explanation = explainer.explain_instance(img.astype('double'), model.predict, hide_color=0, num_samples=20)
    _, lime_mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=False)
    model.layers[-1].activation = None
    grad_mask = discretize_mask(resize_mask(make_gradcam_heatmap(img.reshape(1,28,28), model, last_conv), (28,28)))
    model.layers[-1].activation = tf.keras.activations.softmax
    return lime_mask, grad_mask