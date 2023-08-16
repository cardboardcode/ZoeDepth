import torch
from zoedepth.utils.misc import get_image_from_url, colorize
from PIL import Image
import matplotlib.pyplot as plt


zoe = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)

zoe = zoe.to('cuda')

img = Image.open("images/test.png")
depth = zoe.infer_pil(img)


colored_depth = colorize(depth)
fig, axs = plt.subplots(1,2, figsize=(15,7))
for ax, im, title in zip(axs, [img, colored_depth], ['Input', 'Predicted Depth']):
  ax.imshow(im)
  ax.axis('off')
  ax.set_title(title)

plt.savefig("output.png")
im = Image.fromarray(colored_depth)
im.save("just_depth.png")
     
