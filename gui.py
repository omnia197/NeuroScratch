import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from neuralNetwork import NeuralNetwork
from activation import ReLU, Softmax

model = NeuralNetwork()
model.add(784, 128, ReLU())
model.add(128, 10, Softmax())
model.load("mnist_model.pkl")  

canvas_size = 280
brush_size = 10
pil_image = Image.new("L", (canvas_size, canvas_size), color=255)
draw = ImageDraw.Draw(pil_image)

def predict_digit():
    img = pil_image.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img).astype('float32') / 255.0
    img = img.reshape(1, -1)
    out = model.forward(img)
    pred = np.argmax(out)
    result_label.config(text=f"Prediction: {pred}")

def paint(event):
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=255)
root = tk.Tk()
root.title("Draw a Digit")
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
canvas.pack()
canvas.bind("<B1-Motion>", paint)
btn_predict = tk.Button(root, text="Predict", command=predict_digit)
btn_predict.pack(pady=5)
btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack()
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
result_label.pack()
root.mainloop()
