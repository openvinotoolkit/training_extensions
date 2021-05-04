import torchvision.models.densenet as modelzoo
import torch
import torch.nn as nn

def get_size(blocks,growth_rate,initial_channels):
    
    depth = 5 + 2*sum(blocks)
    l1,l2,l3,l4 = blocks
    width = growth_rate*l4 + (growth_rate*l3+(growth_rate*l2+(growth_rate*l1+initial_channels)//2)//2)//2
    
    return depth, int(width)

def new_lengths(alpha,l1 = 6,l2 = 12,l3 = 24,l4 = 16):
    l1 = int(alpha*l1)
    l2 = int(alpha*l2)
    l3 = int(alpha*l3)
    l4 = int(alpha*l4)
    
    depth = 5+2*(l1+l2+l3+l4)
#     l1 += int((121*alpha - depth)/2) #Uncomment for more accurate depth
#     depth = 5+2*(l1+l2+l3+l4) #Uncomment with the previous line
    
    return depth,[l1,l2,l3,l4]

def new_channels(block_new,alpha,beta,growth_rate=32, init_chan=64,bn_size = 4):
    l1,l2,l3,l4 = block_new
    growth_rate = round(beta/alpha*growth_rate)
    gw = growth_rate
    bn_size = round(alpha*bn_size)
    if bn_size < 2:
        print(f"The initial bn_size{bn_size} needs to be greater than 2.")
        init_chan = 2
    width_temp = gw*l4 + (gw*l3+(gw*l2+(gw*l1)//2)//2)//2
    init_chan = int(8*(1024*beta - width_temp))
    while(init_chan<32):
        print(f"The initial number of channels{init_chan} needs to be greater than 32. Changing growth_rate.")
        growth_rate -= 1
        gw = growth_rate
        width_temp = gw*l4 + (gw*l3+(gw*l2+(gw*l1)//2)//2)//2
        init_chan = int(8*(1024*beta - width_temp))
    width = int(width_temp + init_chan/8)
    return width,growth_rate,init_chan,bn_size

def get_best_values(alpha,beta):
    """
        Alpha changes the depth while beta changes the width
    """
    depth,block_new = new_lengths(alpha)
    width, growth_rate, init_chan,bn_size = new_channels(block_new,alpha,beta)
    return block_new, growth_rate, init_chan, bn_size

def give_model(alpha,beta,num_class = 14,input_shape=(320,320),verbose = 1):
    blocks,gw,b,bn_size = get_best_values(alpha,beta)
    print(blocks,gw,b,bn_size)
    model = modelzoo.DenseNet(growth_rate=gw,block_config=blocks,num_init_features=b,bn_size=bn_size,num_classes=num_class)
    total_mac = densenet_flops(input_shape,blocks,gw,bn_size,b,verbose = verbose)
    return model, total_mac

def last_function(alpha,beta,get_val = False):
    blocks,gw,b,bn_size = get_best_values(alpha,beta)
    print(blocks,gw,b,bn_size)
    model = modelzoo.DenseNet(growth_rate=gw,block_config=blocks,num_init_features=b,bn_size=bn_size,num_classes=14)
    if get_val:
        return get_flops(model,get_val)
    else:
        get_flops(model)

def conv2d_flops(out1,out2,co,cin,k):
    return out1*out2*co*cin*k*k

def batchnorm_flops(out1,out2,c):
    return 2*out1*out2*c

def relu_flops(out1,out2,c):
    return out1*out2*c

def maxpool2d_flops(out1,out2,co,k):
    return out1*out2*co*(k**2-1)

def avgpool2d_flops(out1,out2,co,k):
    return out1*out2*co*(k**2)

def dense_layer_flops(out1,out2,input_layers,growth_rate,k):
    total = 0
    total += batchnorm_flops(out1,out2,input_layers)
    total += relu_flops(out1,out2,input_layers)
    total += conv2d_flops(out1,out2,input_layers,growth_rate*k,1)
    cout = growth_rate*k
    total += batchnorm_flops(out1,out2,cout)
    total += relu_flops(out1,out2,cout)
    total += conv2d_flops(out1,out2,growth_rate,cout,3)
    return total

def dense_block_flops(out1,out2,layers,input_layers,growth_rate,k):
    total = 0
    for layer in range(layers):
        in_layers = input_layers + layer*growth_rate
        total += dense_layer_flops(out1,out2,in_layers,growth_rate,k)

    return total

def transition_flops(out1,out2,input_channels):
    total = batchnorm_flops(out1,out2,input_channels)
    total += relu_flops(out1,out2,input_channels)
    total += conv2d_flops(out1,out2,input_channels,input_channels//2,k=1)
    total += avgpool2d_flops(out1//2,out2//2,input_channels//2,k=2)
    return total

def densenet_flops(input_shape,block_config,growth_rate,k,init_features,verbose = 1):
    out1 = (input_shape[0]-7+2*3)//2 + 1
    out2 = (input_shape[1] -1)//2 + 1
    x = conv2d_flops(out1,out2,init_features,3,7)
    total = x
    if verbose:
        print(f"In first conv, there are {x} flops.")
    c = init_features
    x = batchnorm_flops(out1,out2,c)
    total += x
    if verbose:
        print(f"In batch_norm, there are {x} flops.")
    x = relu_flops(out1,out2,c)
    total += x
    if verbose:
        print(f"In relu, there are {x} flops.")
    out1 = (out1-1)//2 + 1
    out2 = (out2-1)//2 + 1
    total += maxpool2d_flops(out1,out2,c,3)
    if verbose:
        print(f"In maxpool2d, there are {x} flops.")
    for i in range(4):
        if verbose:print(f"At layer{i}, the value of channels:{c} and image:{out1},{out2}.")
        x = dense_block_flops(out1,out2,block_config[i],c,growth_rate,k)
        total += x
        c = block_config[i]*growth_rate + c
        if verbose:
            print(f"In dense_block{i+1}, there are {x} flops.")
    
        if i !=3:
            x = transition_flops(out1,out2,c)
            total += x
            if verbose:
                print(f"In transition_{i+1}, there are {x} flops.")
    
            c = c//2
            out1 = (out1-2)//2 + 1
            out2 = (out2-2)//2 + 1
    x = batchnorm_flops(out1,out2,c)
    total += x
    if verbose:
        print(f"In final_batch norm, there are {x} flops.")
    x = relu_flops(out1,out2,c)
    total += x
    if verbose:
        print(f"In final relu, there are {x} flops.")
    x = c*out1*out2
    total += x
    if verbose:
        print(f"In final average_pooling, there are {x} flops.")
    x = (c+1)*14
    total += x
    if verbose:
        print(f"In final classification, there are {x} flops.")
    return total

