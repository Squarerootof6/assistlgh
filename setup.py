from setuptools import setup, find_packages
import os
import site
import subprocess
# 读取 README 文件内容
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
def get_package_path():
    """ 获取 assistlgh 包的安装路径 """
    for path in site.getsitepackages():
        potential_path = os.path.join(path, "assistlgh")
        if os.path.exists(potential_path):
            return potential_path
    return None
def post_install():
    """ 安装完成后，自动向 ~/.bashrc 添加 alias """
    package_path = get_package_path()
    if package_path:
        alias_command = f"alias chat='python {package_path}/chatbot.py $1 $2 $3'\nalias QA='python {package_path}/chatbot.py quick $1 $2'\n"
        bashrc_path = os.path.expanduser("~/.bashrc")
        # 检查 alias 是否已经存在，避免重复添加
        if os.path.exists(bashrc_path):
            with open(bashrc_path, "r", encoding="utf-8") as f:
                content = f.read()
            if alias_command.strip() not in content:
                with open(bashrc_path, "a", encoding="utf-8") as f:
                    f.write("\n" + alias_command)
                print("Alias added to ~/.bashrc. Run 'source ~/.bashrc' to apply.")
        else:
            with open(bashrc_path, "w", encoding="utf-8") as f:
                f.write(alias_command)
            print("Created ~/.bashrc and added alias.")
        try:
            subprocess.run(["bash", "-c", "source ~/.bashrc"], check=True)
            print("Applied changes with 'source ~/.bashrc'.")
        except Exception:
            print("Please run 'source ~/.bashrc' manually to apply changes.")
    else:
        print("Warning: Could not find assistlgh package path. Alias not added.")
setup(
    name="assistlgh",  # 你的包名称
    version="1.0.0",  # 版本号
    author="Genghao Liu",
    author_email="Squarerootof6@outlook.com",
    description="package serves for Genghao Liu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:Squarerootof6/assistlgh.git",  # 项目主页
    packages=find_packages(),  # 自动查找包
    include_package_data=True,  # 包含 MANIFEST.in 中指定的文件
    package_data={
       'assistlgh':['./*.txt','./arepo_processing/*','assistlgh.mplstyle'], # 指定要包含在包中的额外文件
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=[
        "matplotlib>=3.0",  # 列出依赖项
        "openai",
        "numpy<=1.25",
    ],
    
)
#post_install()