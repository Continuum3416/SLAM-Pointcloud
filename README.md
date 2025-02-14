# ORB SLAM3 Workspace with YOLOv11s:

## Install YOLOv11:

### Linux setup:

YOLO is notorious for its package conflict, so to install it properly, follow these steps: 

<p>First, we need to force pip to python 3 because Linux often uses python2 by default:</p>

```
pip3 install --upgrade --force pip
```

<p>then simply install YOLO by:</p>

```
pip install ultralytics
```

<p>after you install, run it with a test code, if it doesn't work, then there might be some additional pakages that you need:</p>


```

pip install --upgrade importlib_metadata

pip install --upgrade setuptools
```

<p>If you're using a Virtual Machine, make sure to connect to external device (camera before running)</p>

### Window Setup

<p>For windows, use anaconda to install YOLO.</p>


## For ORB SLAM3:

<p>Install ORB SLAM3 at <a href="https://github.com/UZ-SLAMLab/ORB_SLAM3">https://github.com/UZ-SLAMLab/ORB_SLAM3</a> </p>
