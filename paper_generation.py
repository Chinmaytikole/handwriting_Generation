from PIL import Image, ImageDraw

# Create a blank white canvas
width, height = 1979, 2800
canvas = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(canvas)

# Draw horizontal lines 80 pixels apart, starting from y=80 until the second last line
y = 80
i=0
color=''
w = 2
while y < height - 80:

    if i==0 or i==1:
        color = 'white'
    else:
        color='black'
    if y == 240:
        w = 3
        draw.line([(0, y-10), (width, y-10)], fill=color, width=w)
    else:
        w= 2
    draw.line([(0, y), (width, y)], fill=color, width=w)
    y += 80
    i +=1

# Draw one vertical line, starting 80 pixels from the left (x=80), then at intervals of 80 pixels 3 times
x = 0
for i in range(3):
    x += 80
    if(i==0 or i==1):
        continue
    draw.line([(x, 0), (x, height)], fill="black", width=2)
    x += 80  # Move the line 80 pixels to the right for the next line

# Save the image
canvas.save("canvas_with_lines.png")
canvas.show()
