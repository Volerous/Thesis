#%%
import glob
from PIL import Image
import PIL
#%%
images = glob.glob("../images/heart-pet/image-*-.jpg")
images.sort()
hearts = [Image.open(i) for i in images]
# for i in hearts:
#     i = i.resize((i.width *2, i.height*2), PIL.Image.NEAREST)
#%%
hearts[0].save("../images/heart-pet/heart-rotation.gif",
            append_images=hearts[1:],
            save_all=True,
            duration=150,
            # version="GIF89a",
            loop=0)
#%%
images = glob.glob("../images/heart-pet/image-*-.jpg")
images.sort()
ims = [Image.open(i) for i in images]
#%%
ims[0].save("../images/heart-pet/heart-rotation.gif",
            append_images=ims[1:],
            save_all=True,
            duration=500,
            version="GIF89a",
            loop=0)
