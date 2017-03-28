from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules=[
    Extension("cy_yolo2_findboxes",
              sources=["cy_yolo2_findboxes.pyx"],
              libraries=["m"] # Unix-like specific
    )
]



setup(

    name= 'cy_yolo2_findboxes',
    ext_modules = cythonize(ext_modules),
)