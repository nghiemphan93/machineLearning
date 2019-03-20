from tkinter import *





x = 10
y = 50
r = 40
def animation():
   global x, y
   c.delete("all")
   c.create_oval(x-r, y-r, x+r, y+r, fill="orange")
   x = x + 1
   c.after(int(1000/60), animation)
root = Tk()
c = Canvas(root, height=400, width=400)
c.pack()
animation()
root.mainloop()