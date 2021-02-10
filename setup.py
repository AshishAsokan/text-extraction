import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "text-extraction",
    version = "1.0",
    author = "Ashish Ramayee Asokan",
    author_email = "raashish020@gmail.com",
    description = "Extracting Text from Videos using OpenCV",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/AshishAsokan/text-extraction.git",
    packages = setuptools.find_packages(),
    install_requires = [
        'scipy==1.4.1',
        'imutils==0.5.4',
        'pytesseract==0.3.7',
        'numpy==1.16.2',
        'opencv_python==4.2.0.34',
        'scikit_image==0.16.2',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
)