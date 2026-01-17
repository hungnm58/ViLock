#!/usr/bin/env python3
"""Generate menu bar icon for AutoLock."""

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow required: pip install Pillow")
    exit(1)

# Create 18x18 template icon (standard menu bar size)
size = 18
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Draw a simple lock shape
# Body
draw.rectangle([4, 8, 14, 16], fill='black')
# Shackle
draw.arc([5, 2, 13, 10], 0, 180, fill='black', width=2)

img.save('icon.png')
print("Created icon.png (18x18)")

# Create 2x version for retina
size2x = 36
img2x = Image.new('RGBA', (size2x, size2x), (0, 0, 0, 0))
draw2x = ImageDraw.Draw(img2x)

draw2x.rectangle([8, 16, 28, 32], fill='black')
draw2x.arc([10, 4, 26, 20], 0, 180, fill='black', width=4)

img2x.save('icon@2x.png')
print("Created icon@2x.png (36x36)")
