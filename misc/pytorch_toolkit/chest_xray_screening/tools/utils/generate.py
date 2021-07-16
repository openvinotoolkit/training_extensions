import torchvision.models.densenet as modelzoo

MIN_CHANNELS = 32 # The minimum number of channels
CHANNEL_FACTOR = 8 # Factor at which the number of channels should be changed.
def get_size(blocks, growth_rate, initial_channels):

    depth = 5 + 2 * sum(blocks)
    # Here 5 is the minimum depth of CNN and 2 is the factor at which the depth should be changed.
    # Choice of value 5 and 2 is arbitrary, can be modifed.
    l1, l2, l3, l4 = blocks
    width = growth_rate * l4 + (
        growth_rate * l3 + (
            growth_rate * l2 + (
                growth_rate * l1 + initial_channels) // 2) // 2) // 2

    return depth, int(width)


def new_lengths(alpha, l1=6, l2=12, l3=24, l4=16):
    l1 = int(alpha * l1)
    l2 = int(alpha * l2)
    l3 = int(alpha * l3)
    l4 = int(alpha * l4)

    depth = 5 + 2 * (l1 + l2 + l3 + l4) 
    # Here 5 is the minimum depth of CNN and 2 is the factor at which the depth should be changed.
    # Choice of value 5 and 2 is arbitrary, can be modifed.

#     l1 += int((121*alpha - depth)/2) # Uncomment for more accurate depth
#     depth = 5+2*(l1+l2+l3+l4) # Uncomment with the previous line

    return depth, [l1, l2, l3, l4]


def new_channels(block_new, alpha, beta, growth_rate=32,
                 init_chan=64, bn_size=4):
    l1, l2, l3, l4 = block_new
    growth_rate = round(beta / alpha * growth_rate)
    gw = growth_rate
    bn_size = round(alpha * bn_size)
    if bn_size < 2:
        print(f"The initial bn_size {bn_size} needs to be greater than 2.")

    width_temp = gw * l4 + (gw * l3 + (gw * l2 + (gw * l1) // 2) // 2) // 2
    init_chan = int(CHANNEL_FACTOR * (1024 * beta - width_temp))
    while init_chan < MIN_CHANNELS:
        print(f"The initial number of channels {init_chan} needs to be greater than 32. Changing growth_rate.")

        growth_rate -= 1
        gw = growth_rate
        width_temp = gw * l4 + (gw * l3 + (gw * l2 + (gw * l1) // 2) // 2) // 2
        init_chan = int(CHANNEL_FACTOR * (1024 * beta - width_temp))
    width = int(width_temp + init_chan / CHANNEL_FACTOR)
    return width, growth_rate, init_chan, bn_size


def get_best_values(alpha, beta):
    """
        Alpha changes the depth while beta changes the width
    """
    _, block_new = new_lengths(alpha)
    _, growth_rate, init_chan, bn_size = new_channels(block_new, alpha, beta)
    return block_new, growth_rate, init_chan, bn_size


def give_model(alpha, beta, num_class=3, input_shape=(320, 320), verbose=1):
    blocks, gw, b, bn_size = get_best_values(alpha, beta)
    print(blocks, gw, b, bn_size)
    model = modelzoo.DenseNet(
        growth_rate=gw,
        block_config=blocks,
        num_init_features=b,
        bn_size=bn_size,
        num_classes=num_class)

    return model
