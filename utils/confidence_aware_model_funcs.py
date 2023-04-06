import numpy as np
import torch
from PIL import Image

def adjust_fft_amplitude(main_image:torch.Tensor, adjusting_image:torch.Tensor, mask:torch.Tensor, alpha:float = 1, beta:float = 1)->torch.Tensor:
    """
    If A is amplitude specter of main_image and A' is amplitude specter of adjusting_image, then the formula given in the paper
    (Enhancing Pseudo Label Quality for Semi-Supervised Domain-Generalized Medical Image Segmentation, 2022)
    is following:

    A_new = A*(1-alpha)*(1-mask) + A'* alpha * mask

    Notice that only amplitude is adjusted, not the phase, thus the structure of image will not be changed
    """
    amplitude = lambda ff: torch.sqrt(ff.real**2 + ff.imag**2)
    
    ff_main_image = torch.fft.fft2(main_image)
    ff_adjusting_image = torch.fft.fft2(adjusting_image)

    new_amplitude_multiplier = amplitude(ff_adjusting_image)/amplitude(ff_main_image)

    assert alpha > 0, "Alpha has to be bigger than 0"
    assert beta > 0, "Alpha has to be bigger than 0"
    ff_adjusted_image = ff_main_image * (abs(mask - 1)) * alpha + ff_main_image * beta * mask * new_amplitude_multiplier

    return torch.fft.ifft2(ff_adjusted_image)



def confidence_aware_unsupervised_loss(O_prediction1, F_prediction1, O_prediction2, F_prediction2, beta:float = 1):
    def KL_divergence(O_prediction, F_prediction):
        return torch.nanmean(F_prediction * torch.log(F_prediction/O_prediction))

    def unsupervised_loss(prediction, ground_truth, variance):
        """
        Variance tells us how close those two predictions were.
        If they were too different, the variance is large thus encreasing 
        the whole loss but decresing the part from cross entropy as the ground 
        truth is not accurate
        """
        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(prediction, ground_truth)
        return torch.exp(-variance) * loss  + variance
    
    variance1 = KL_divergence(O_prediction1, F_prediction1)
    variance2 = KL_divergence(O_prediction2, F_prediction2)
    print('variance1', variance1)
    print('variance2', variance2)

    combined_prediction1 = (O_prediction1 + F_prediction1)/2
    combined_prediction2 = (O_prediction2 + F_prediction2)/2

    # dim = 1 should be over classes - dim 0 is batch number and then are dimensions of inputs
    seg_map1 = torch.argmax(combined_prediction1, dim = 1) 
    seg_map2 = torch.argmax(combined_prediction2, dim = 1)

    loss1 = unsupervised_loss(combined_prediction1, seg_map2, variance2)
    loss2 = unsupervised_loss(combined_prediction2, seg_map1, variance1)

    return beta * (loss1 + loss2)



if __name__ == "__main__":
    pil_image1 = Image.open("utils/3.png")
    pil_image2 = Image.open("utils/5.png")

    image1 = torch.from_numpy(np.array(pil_image1.getdata()).reshape(pil_image1.size[1], pil_image1.size[0], 3))
    image2 = torch.from_numpy(np.array(pil_image2.getdata()).reshape(pil_image2.size[1], pil_image2.size[0], 3))

    image1 = image1[:,:,0]
    image2 = image2[:,:,0]

    mask1 = (image2 >0.5).type(torch.uint8)
    mask2 = (image1 >0.5).type(torch.uint8)
    # print(mask1)

    print(confidence_aware_unsupervised_loss(mask1, mask1, mask2, mask2))


    

    # mask = torch.zeros_like(image2)
    # mask[23:823,64:1024] = 1
    # # mask = abs(mask-1)

    # output = adjust_fft_amplitude(image1, image2, mask)

    # image = Image.fromarray(np.uint8(amplitude(output).numpy()))
    # image.save("Fourier adjustement.png")
