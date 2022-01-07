import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import utils as utils
import torchinfo as torchinfo
import os
from torchvision.io import read_image
import numpy as np
import torch.nn.functional as F

DEVICE = 'cpu'


# def apply_softmax(image, mask):
# TO-DO 1
def sigmoid(x):
    '''
    Implement the sigmoid function using the formula given in the proj2 notebook. 
    You are not allowed to use any pre-defined sigmoid function from any library. 
    
    Args:
        x: N x 1 torch.FloatTensor
        
    Returns:
        y: N x 1 torch.FloatTensor
    '''
    y = []
    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################   
    x = torch.stack([x], dim = 1)
    output = torch.empty(len(x))
    for index in range(len(x)):
        output[index] = 1 / (1 + torch.exp(-x[index]))
    y = output    
    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return y

# TO-DO 2
def softmax(x):
    '''
    Implement the softmax function using the formula given in the proj2 notebook. 
    You are not allowed to use any pre-defined softmax function from any library. 
    
    Args:
        x: N x 1 torch.FloatTensor
        
    Returns:
        y: N x 1 torch.FloatTensor
    '''
    y = []
    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    x = torch.stack([x], dim = 1)
    softmaxx = torch.empty(len(x))
    expi = torch.empty(len(x))
    for index in range(len(x)):
        expi[index] = torch.exp(x[index])
    summ_expi = torch.sum(expi)
    for jindex in range(len(x)):
        softmaxx[jindex] = (torch.exp(x[jindex]))/summ_expi
    y = softmaxx
    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return y

# TO-DO 3
def apply_mask_to_image(image, mask):
    '''
    Applies segmentation mask of image to the original image to produce the final segmented image. 
    
    For this function, we will be applying the color red to the final segmented image.
    
    You will need to implement these steps:
    1.) Make `mask` the same shape as `image` such that each channel in the resulting `colored_mask` is equal to `mask`.
    2.) Perform broadcast multiplication on `mask` with the red color list in the `colors` dictionary. 
        The resulting `colored_mask` should contain values different from 0 in the red channel of the mask ONLY. 
    3.) Combine `image` and `colored_mask` by performing element-wise summation, and return the result as type integer.
    4.) Make sure all values in the `final_seg_img` fall between 0 and 255.
    
    '''
    image = torch.from_numpy(image).type(torch.int64)
    mask = torch.from_numpy(mask).type(torch.int64)
    colors = {"red": torch.Tensor([255,0,0])}
      
    colored_mask = [] 
    final_seg_img = []

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    red = colors.get('red').type(torch.int64)
    rgb_channel = mask.repeat(1,1,3) 
    
    rgb_channel = rgb_channel.mul(red)
    colored_mask = rgb_channel
   
    final_seg_image = ((colored_mask + image).clamp(0,255))
    final_seg_img = final_seg_image.type(torch.int64)
    
    
    
    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    
    return final_seg_img


# TO-DO 4
def load_FPN_resnet50():
    '''
    Repeat the steps at the beginning of Section 2.2: VGG-19 for the ResNet-50 model to see 
    how well it performs in the cell below. 
    
    You will keep the same `classes`, and `activation` function. But, you will need to change 
    the `encoder_name` to `"resnet50"` to load the ResNet-50 model from `segmentation_models_pytorch`. 
    
    Please see the `models/` directory for the appropriate saved weights file for the ResNet-50 model.
    '''
    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    resnet50_model = smp.FPN(encoder_name='resnet50',classes=1,activation='sigmoid')
    resnet50_model.load_state_dict(torch.load('./models/resnet50_best_model_weights.pt', map_location=torch.device('cpu')))

    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return resnet50_model

# TO-DO 5
def IoU(predict: torch.Tensor, target: torch.Tensor):

    """Compute IoU on torch tensor
    
    Args:
        predict: MxN torch tensor represeting predicted label map,
            each value in range 0 to 1.
        target: MxN torch tensor representing ground truth label map,
            each value in range 0 to 1.

    Returns: Float/Double that represent the IoU score
    """
    
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    #raise NotImplementedError('`IoU()` function in ' +
    #'`student_code.py` needs to be implemented')
   

    #i = (predict & target).sum()
    #u = (predict | target).sum()
    #x = (i/u)
    #IoU = x
    
    u = torch.sum(((predict + target) > 0)*1)
    i = torch.sum(((predict*target))*1)
    IoU = i/u.item()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return IoU

# TO-DO 6
def applyIoU(model: smp.fpn.model, dataset: utils.Dataset):
    
    """
    Apply the IoU function you wrote above to evaluate the performance 
    of the input model on the input dataset.
    
    Hint: true_mask is a np.ndarray. You will need to convert it to torch.Tensor; 
    otherwise, you will get an AssertionError.
    
    Args: 
        model: the smp model that you will evaluate
        dataset: the dataset that model will be tested on; 
                 It contains N (image, true mask) pair
    Returns:
        IoU_score: N Iou scores, one score corresponding to one pair
        
    """
    
    IoU_score = []
    
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    #raise NotImplementedError('`applyIoU()` function in ' +
    #'`student_code.py` needs to be implemented')
    #IoU_score = torch.zeros(size=(len(dataset),1))
    iteration = 0
    length = len(dataset)
    IoU_score = [0]*length
    for i in range(len(dataset)):
        image,true_mask = dataset[i]
        x_tensor = torch.from_numpy(image).to('cpu').unsqueeze(0)
        pred_mask = model.predict(x_tensor)
        pred_mask = pred_mask.numpy()
        pred_mask = (pred_mask > 0.5)*1 #needs to be done to convert values to 0 and 1
        pred_mask = torch.from_numpy(pred_mask)
        true_mask = torch.from_numpy(true_mask).to('cpu').unsqueeze(0)
        IoU_score[i] = IoU(pred_mask, true_mask)
        iteration = 1 + iteration
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    assert type(true_mask) == torch.Tensor
    
    return IoU_score

# TO-DO 7
def compare_psp_fpn(test_dataset):
    """
    Step 1: get FPN segmentation model with resnet50 from above
    Step 2: create PSP segmentation model with pretrained resnet50 for PSP
    Step 3: Get the IoU score 
    
    Args:
        x_dir, y_dir: the input for FCN-ResNet50 function
        test_dataset: the dataset that we will use to find out IoU score
        
    Returns:
        psp_iou: the IoU score for the PSPNet with Resnet50 
        fpn_iou: the IoU score for the FPN with Resnet50
        
    
    """
    
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    # Step 1: Load FPN + ResNet50
    resnet50_model = smp.FPN(encoder_name='resnet50',classes=1,activation='sigmoid')
    resnet50_model.load_state_dict(torch.load('./models/resnet50_best_model_weights.pt', map_location=torch.device('cpu')))
    
    # Step 2: create PSP Segmentation model with pretrained resnet50 
    psp_model = smp.PSPNet(encoder_name='resnet50',classes=1,activation='sigmoid')
    psp_model.load_state_dict(torch.load('./models/pspnet_resnet50_best_model_weights.pt', map_location=torch.device('cpu')))
    
    # Step 3: IoU Scores
    psp_iou = []
    fpn_iou = []
    
    fpn_iou = applyIoU(resnet50_model, test_dataset)
    psp_iou = applyIoU(psp_model, test_dataset)
   
    #raise NotImplementedError('`compare_psp_fpn()` function in ' +
    #'`student_code.py` needs to be implemented')
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return psp_iou, fpn_iou

def load_model(decoder_weights_path=None):
    ENCODER = 'resnet50' 
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['car']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    model.load_state_dict(torch.load('models/{}_best_model_weights.pt'.format(ENCODER), map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(decoder_weights_path, map_location=torch.device('cpu')))
    return model

def print_model_summary(model, channels=3, H=384, W=480):
    print(torchinfo.summary(model, input_size=(1, channels, H, W)))
    

def create_vis_dataset():
    # should paths and classes be defined here or in notebook?
    x_vis_dir = "./data/CamVid/test/"
    y_vis_dir = "./data/CamVid/testannot/"
    classes = ["car"]
    return utils.Dataset(x_vis_dir, y_vis_dir, classes)

def create_test_dataset():

    x_test_dir = "./data/CamVid/test/"
    y_test_dir = "./data/CamVid/testannot/"
    classes = ["car"]
    ENCODER = 'resnet50' 
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    test_dataset = utils.Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=utils.get_validation_augmentation(), 
        preprocessing=utils.get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    return test_dataset

def test_model(model, test_dataset, vis_dataset):

    for i in range(len(test_dataset)):

        image_vis = vis_dataset[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        print(x_tensor.shape)
        pr_mask = model.predict(x_tensor)
        print(f"Shape of predicted mask = {pr_mask.shape}")
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        utils.visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )
    return image, pr_mask
