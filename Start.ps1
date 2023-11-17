# 手动初始化 Conda 环境，此处修改为你的anaconda安装位置
$AnacondaPath = "C:\Program Files\Anaconda"
& "$AnacondaPath\Scripts\conda.exe" 'shell.powershell' 'hook' | Out-String | Invoke-Expression

# 激活 Anaconda 环境
conda activate xuexi

# 切换到脚本所在的目录
cd “D:\Desktop”

# 执行 Python 脚本
python ceshi.py