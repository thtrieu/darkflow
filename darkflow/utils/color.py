def rgb2hex(r, g, b, t=255):
		r = max(0, min(int(r), 255))
		g = max(0, min(int(g), 255))
		b = max(0, min(int(b), 255))
		t = max(0, min(int(t), 255))
		return '#%02x%02x%02x%02x' % (r, g, b, t)
