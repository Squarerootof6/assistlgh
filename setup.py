from setuptools import setup, find_packages

# 读取 README 文件内容
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
       'assistlgh':['./*.txt','./arepo_processing/*'], # 指定要包含在包中的额外文件
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=[
        "matplotlib>=3.0",  # 列出依赖项
    ],
)
