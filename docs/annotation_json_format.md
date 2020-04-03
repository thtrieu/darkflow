# using json as an annotation format

previously, darkflow only supported pascal-voc xml format. JSON seemed to be another good option as an annotation format and a parser for json format has been added.

## how to use json parser during training

`--annotationformat` option has been added to the `flow` command. If it is not specified or the user gives `xml` as a value, then it will use the pascal-voc xml parser.

On the other hand, if the user gives `json` as a value, then it will utilize the json parser.

## json format

The json parser will parse the files inside the specified annotation dir according to the following format.

- `imgfile`: the name of the image file which should be inside the specified `--images` dir
- `w`: the width of the image
- `h`: the height of the image
- `objects`: json array of the object info. Each object should have the following key-values:
    - `rect`: json object which contains the 4 coordinates that specify the bounding box
        - `y1`: one value of box's height
        - `y2`: the other value of the box's height
        - `x1`: one value of the box's width
        - `x2`: the other value of the box's width
    - `name`: the label for this box

Here is an example:
```
{"imgfile": "0313.png", "w": 640, "h": 480, "objects": [{"rect": {"y1": 4, "y2": 144, "x1": 385, "x2": 587}, "name": "face"}]}
```

# Notice

the json parser will check the x1,x2 / y1,y2 comparison and it will correct it.