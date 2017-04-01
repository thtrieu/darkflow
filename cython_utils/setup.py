from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

if os.name =='nt' :
  
    ext_modules=[
        Extension("nms",
                sources=["nms.pyx"],
                #libraries=["m"] # Unix-like specific
                    include_dirs=[numpy.get_include()]
        )
    ]

    ext_modules_yolo2=[
        Extension("cy_yolo2_findboxes",
                  sources=["cy_yolo2_findboxes.pyx"],
                  #libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
        )
    ]

    ext_modules_yolo=[
        Extension("cy_yolo_findboxes",
                  sources=["cy_yolo_findboxes.pyx"],
                  #libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
        )
    ]

else :
    
    ext_modules=[
        Extension("nms",
                sources=["nms.pyx"],
                libraries=["m"] # Unix-like specific
        )
    ]

    ext_modules_yolo2=[
        Extension("cy_yolo2_findboxes",
                  sources=["cy_yolo2_findboxes.pyx"],
                  libraries=["m"] # Unix-like specific
        )
    ]

    ext_modules_yolo=[
        Extension("cy_yolo_findboxes",
                  sources=["cy_yolo_findboxes.pyx"],
                  libraries=["m"] # Unix-like specific
        )
    ]




setup(

    #name= 'cy_findboxes',
    ext_modules = cythonize(ext_modules),
)

setup(

    #name= 'cy_findboxes',
    ext_modules = cythonize(ext_modules_yolo2),
)



setup(

    #name= 'cy_findboxes',
    ext_modules = cythonize(ext_modules_yolo),
)