from torch import nn, float32, Tensor as T
import torchvision as tv
import torch
#import torchsummary as ts

import numpy as np

def conv2d_output_size(layer: nn.Conv2d, input_size: tuple):

    num_fms = layer.out_channels
    fm_width = ((input_size[1] + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0]) + 1
    fm_height = ((input_size[2] + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1]) + 1

    return (num_fms, int(fm_width), int(fm_height))

# def conv2d_output_size(layer: nn.Conv2d, input_size: tuple):


#     num_fms = layer.out_channels
#     fm_width = ((input_size[1] + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0]) + 1
#     fm_height = ((input_size[2] + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1]) + 1

#     return (num_fms, int(fm_width), int(fm_height))


def conv2d_as_gemm(layer: nn.Conv2d, input: torch.Tensor) -> torch.Tensor:
    # transpose = False
    w = layer.weight
    w_flat = w.flatten(1, 3).detach().numpy()
    
    input_gemm = nn.functional.unfold(input=input, kernel_size=layer.kernel_size, dilation=layer.dilation, padding=layer.padding, stride=layer.stride)
    input_gemm = input_gemm.detach().numpy()
    # if(input_gemm.shape[0] != w_flat.shape[1]):
    #     transpose = True
    #     w_flat = w_flat.T

    print(f"\t weight tensor size = {w.shape}, flattened = {w_flat.shape}, dtype = {w.dtype}")
    print(f"\t input size = {input.shape}, gemm size = {input_gemm.shape}")

    out = (w_flat @ input_gemm) # Matrix mult

    # if(transpose):
    #     out = out.T

    out_tens = torch.from_numpy(out)
    out_tens_size = conv2d_output_size(layer, input.shape)
    out_tens = nn.functional.fold(out_tens, output_size=out_tens_size[1:], kernel_size=(1, 1))
    print(f"\t output gemm size = {out.shape}, folded size = {out_tens.shape}")
    print(f"\t VITIS (M, N, K) = {w_flat.shape[0]}, {w_flat.shape[1]}, {input_gemm.shape[1]}")

    return out_tens


def linear_as_gemm(layer: nn.Linear, input: torch.Tensor) -> torch.Tensor:

    raise NotImplementedError ("Linear layer to GEMM calculation has not been implemented yet")

    # # transpose = False
    # w = layer.weight
    # print(w.shape)
    # w_flat = w.detach().numpy()
    
    # input_gemm = nn.functional.unfold(input=input, kernel_size=(1, 1))
    # input_gemm = input_gemm.detach().numpy()
    # # if(input_gemm.shape[0] != w_flat.shape[1]):
    # #     transpose = True
    # #     w_flat = w_flat.T

    # print(f"\t weight tensor size = {w.shape}, flattened = {w_flat.shape}, dtype = {w.dtype}")
    # print(f"\t input size = {input.shape}, gemm size = {input_gemm.shape}")

    # out = (w_flat @ input_gemm) # Matrix mult

    # # if(transpose):
    # #     out = out.T

    # out_tens = torch.from_numpy(out)
    # out_tens_size = conv2d_output_size(layer, input.shape)
    # out_tens = nn.functional.fold(out_tens, output_size=out_tens_size[1:], kernel_size=(1, 1))
    # print(f"\t output gemm size = {out.shape}, folded size = {out_tens.shape}")

    # return out_tens

ORIG_INPUT_SHAPE = (3, 224, 224)

@torch.no_grad()
def main():

    m = tv.models.resnet34(weights=tv.models.ResNet34_Weights.DEFAULT, progress=True).float()
    input_tens = torch.from_numpy(np.random.rand(ORIG_INPUT_SHAPE[0], ORIG_INPUT_SHAPE[1], ORIG_INPUT_SHAPE[2])).float()




    count_down = -1
    is_down_sample = -2

    for id, l in enumerate(m.modules()):
        if(count_down != -1):
            count_down -= 1
        if(is_down_sample != -2):
            is_down_sample -= 1

        if(id == 0):
            continue
        if isinstance(l, nn.Sequential):
            continue
        if isinstance(l, tv.models.resnet.BasicBlock):
            identity = input_tens
            if l.downsample is None:
                count_down = len(list(l.modules())) - 1
                #print(f"BasicBlock has {count_down - 1} layers in it")
            elif l.downsample is not None:
                is_down_sample = len(list(l.modules())) - 2
                print(f"Basic Block has downsample module in {is_down_sample} layers ")

            continue


        if(count_down == 0):
            #print(f"cd: now on layer {id}")
            if isinstance(l, nn.BatchNorm2d):
                input_tens = input_tens[None, :, :, :]
                out_tens = l.forward(input_tens)
                out_tens = out_tens.squeeze()
                #print(f"\t Batchnorm layer requires 4d. input_tens shape = {input_tens.shape}, output_tens shape = {out_tens.shape}")

                out_tens = out_tens + identity
                out_tens = nn.functional.relu(out_tens, inplace=True)
                input_tens = out_tens

            else:
                raise TypeError (f"This layer should have been a batchnorm2d, but instead it was: {l}")

            continue

        if(is_down_sample == 0 or is_down_sample == -1):
            #print(f"ds: now on layer {id}")
            if(is_down_sample == 0):
                if isinstance(l, nn.Conv2d):
                    print(id, "->", l)
                    new_identity = conv2d_as_gemm(l, identity)
                    
                    check_out = l.forward(identity)
                    err = (check_out - new_identity).abs().max()
                    print(f"\t check tensor size: {check_out.shape} max error between gemm and forward: {err}")
                    identity = new_identity
                else:
                    raise TypeError (f"This layer should have been a conv2d, but instead it was: {l}")
            elif(is_down_sample == -1):
                if isinstance(l, nn.BatchNorm2d):
                    input_tens = input_tens[None, :, :, :]
                    out_tens = l.forward(input_tens)
                    out_tens = out_tens.squeeze()
                    #print(f"\t Batchnorm layer requires 4d. input_tens shape = {input_tens.shape}, output_tens shape = {out_tens.shape}")

                    out_tens = out_tens + identity
                    out_tens = nn.functional.relu(out_tens, inplace=True)
                    input_tens = out_tens
                else:
                    raise TypeError (f"This layer should have been a batchnorm2d, but instead it was: {l}")

            continue


        
        if isinstance(l, nn.Conv2d):
            print(id, "->", l)
    
            out_tens = conv2d_as_gemm(l, input_tens)

            check_out = l.forward(input_tens)
            err = (check_out - out_tens).abs().max()
            print(f"\t check tensor size: {check_out.shape} max error between gemm and forward: {err}")

        elif isinstance(l, nn.Linear):
            
            #print(id, "->", l)
            #print(f"\t input size = {input_tens.shape}")
            out_tens = l.forward(input_tens.squeeze(dim = 2).t())
    
        #     out_tens = linear_as_gemm(l, input_tens)
        #     check_out = l.forward(input_tens)
        #     err = (check_out - out_tens).abs().max()
        #     print(f"\t check tensor size: {check_out.shape} max error between gemm and forward: {err}")            

        elif isinstance(l, nn.BatchNorm2d):
            #print(id, "->", l)
            input_tens = input_tens[None, :, :, :]
            out_tens = l.forward(input_tens)
            out_tens = out_tens.squeeze()
            #print(f"\t Batchnorm layer requires 4d. input_tens shape = {input_tens.shape}, output_tens shape = {out_tens.shape}")

        else:
            #print(id, "->", l)
            out_tens = l.forward(input_tens) # Just do the next layer
        
        input_tens = out_tens

    return

if __name__ == "__main__":
    main()
    exit(0)