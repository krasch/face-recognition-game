Based on https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd

# Installation

Python >= 3.7

```
sudo apt-get install libopenblas-dev liblapack-dev 
pip install -r requirements.txt  
sh download_models.sh
```

# Startup

```
python main.py
```


# Notes

CNN face detection does not properly run in thread on my laptop

# known issues

* thumbnail_at_right_place[y+MATCH_DISPLAY_SIZE: y + MATCH_DISPLAY_SIZE*2, x+MATCH_DISPLAY_SIZE: x + MATCH_DISPLAY_SIZE*2] = thumbnail
 ValueError: could not broadcast input array from shape (100,100,3) into shape (100,49,3)
 when display too many people at once!
* need to store unaligned box
* monitoring in config