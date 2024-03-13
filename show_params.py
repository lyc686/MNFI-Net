from torchvision.models import resnet50
import torch
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
 
# model = models.resnet50()   
checkpoints = './runs/train/ours/weights/best.pt'
model = torch.load(checkpoints)
model_name = 'MNFI-Net'

print(model)
# flops, params = get_model_complexity_info(model, (3, 1024, 1024),as_strings=True,print_per_layer_stat=True)
# print("%s |%s |%s" % (model_name,flops,params))


# from torchvision.models import resnet50
# from thop import profile

# model = torch.load(checkpoints)
# flops, params = profile(model, input_size=(1, 3, 640,640))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')



# from flops_counter import get_model_complexity_info
# import torch
# checkpoints = './runs/train/ours/weights/best.pt'
# net = torch.load(checkpoints)
# flop, param = get_model_complexity_info(net, (3, 1024, 1024), as_strings=True, print_per_layer_stat=False)
# print("GFLOPs: {}".format(flop))
# print("Params: {}".format(param))
