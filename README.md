# 深度学习课程报告

使用`docker`运行容器，命令为：`docker run -itd -v "D:\Projects\DLHW:/rl" --gpus "device=0" --name "DLHW" ubuntu /bin/bash`，在容器内执行`apt install python3 pip cmake zlib1g-dev git libsdl1.2-dev -y`和`pip install -r requirements.txt`完成环境配置，最后执行`python3 /DLHW/run.py -c`训练。

报告附录列出了所需的Python包。
