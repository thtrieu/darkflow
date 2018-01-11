def rgb2hex(r, g, b):
        r = max(0, min(int(r), 255))
        g = max(0, min(int(g), 255))
        b = max(0, min(int(b), 255))
        return '#%02x%02x%02xFF' % (r, g, b)
