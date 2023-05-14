import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad



# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_result_images(gen, latent):
    gen.eval()

    with torch.no_grad():
        generated = gen(latent)

    generated = generated * 0.5 + 0.5

    return generated

def gradient_penalty(disc, real_data, fake_data, device):
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(device)

        prob_interpolated = disc(interpolated)

        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return 10 * ((gradients_norm - 1) ** 2).mean()


