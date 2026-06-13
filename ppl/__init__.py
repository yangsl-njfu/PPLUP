import sys
import os

# 将本地 metadrive 源码加入 Python 路径，避免依赖 pip 安装的版本
_metadrive_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadrive")
if _metadrive_src not in sys.path:
    sys.path.insert(0, _metadrive_src)
del _metadrive_src