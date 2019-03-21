import tkinter as tk
from tkinter import N,S,E,W
from tkinter import ttk
from tkinter import filedialog
import copy
import os
import os.path 
from PIL import ImageTk, Image
from app.tools.util import resize_and_crop, pad_and_resize

from app.classifier import Classifier as Clf
from app.resultviewer import ResultViewer

from functools import partial

class ResultsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.rv = ResultViewer('D:/Downloads/Deep Learning/Week 6', './results/val_set_results.pt')
        
        self.cls = None
        self.img_res = []

        backButton = tk.Button(self, text='Back', command=lambda: controller.show_frame(Main_2))
        backButton.pack(side='top', fill='x')

        self.panel = tk.Frame(self)
        self.panel.pack(side='top', fill='both', expand=True)
        
        self.sideView = tk.Frame(self.panel)
        self.sideView.grid(row=0, column=2,sticky='nesw')

        self.sideView.columnconfigure(0, weight=1)
        self.sideView.columnconfigure(1, weight=1)
        self.sideView.rowconfigure(1, weight=1)

        self.img = tk.Label(self.sideView)
        self.img.grid(row=0, column=0, columnspan=2 , sticky='nsew')

        self.score = tk.Label(self.sideView, text = '')
        self.score.config(font=('Arial',14))
        self.score.grid(row=1, column=0, columnspan=2, sticky='nsew')
        
        
        self.tree = ttk.Treeview(self.panel)
        self.tree.grid(row=0, column=0, sticky='nsw')
        
        scroll = ttk.Scrollbar(self.panel)
        scroll.grid(row=0, column=1, sticky="nsw") 

        scroll.configure(command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)


        self.panel.columnconfigure(2, weight=1)
  

        self.panel.rowconfigure(0, weight=1)

        self.tree.bind("<Double-1>", self.onDoubleClick)

    def reset(self):
        self.tree.delete(*self.tree.get_children(''))
        self.img.config(image='')
        self.img.image = None
        self.score.config(text='')
        self.score.update_idletasks()

    def init_tree(self, cls):
        self.reset()
        self.cls = cls

        self.img_res = self.rv.get_class_results(cls)
     
        img_list = self.img_res.index.values
        
        for i in img_list:
            self.tree.insert("",'end', text=i, value=i)

    def onDoubleClick(self, event):
        item = self.tree.selection()[0]
        img_name = self.tree.item(item,"text")
        img_path = self.rv.get_img_path(img_name)

        score = self.img_res.loc[img_name, self.rv.class_to_index()[self.cls]+1]
        self.score.config(text='Prediction score: {:.2f} %'.format(float(score)*100))
        self.score.update_idletasks()
        selected_img = ImageTk.PhotoImage(pad_and_resize(img_path, 400))
        
        self.img.config(image=selected_img)
        self.img.image = selected_img
        self.img.update_idletasks()

    def show(self, cls):
        self.cls = cls
        self.init_tree(cls)
        self.lift()


class Main_1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.file = None
        self.clf = Clf('./results/pascalvoc_A.pt')

        backButton = tk.Button(self, text='Back', command=lambda: controller.show_frame(MainView))
        backButton.pack(side='top', fill='x')

        self.leftFrame = tk.Frame(self)
        self.leftFrame.pack(side="left", fill="both", expand=True)
        self.rightFrame = tk.Frame(self)
        self.rightFrame.pack(side="right", fill="both", expand=True)

        self.imgPanel = tk.Label(self.leftFrame, text=str(self.file or 'No file uploaded'))
        self.imgPanel.pack(side="left", fill="both", expand=True)

        self.results = tk.Label(self.rightFrame, text='Prediction:\n {}'.format(None), anchor='e')
        self.results.config(font=('Arial',14))
        self.results.pack(side='top', fill='y', expand=True)

        openFile = tk.Button(self.rightFrame, text="Open a Image", command= self.uploadFile)
        openFile.config(bg='#8e8d8d', font=('Arial',14))
        openFile.pack(side='bottom', fill='both', expand=True)

    def predict(self, img_path):
        self.results.config(text='Prediction:\n {}'.format('Calculating!!!'))
        self.results.update_idletasks()
        results = self.clf.predict(img_path)
        results_string = '\n'.join(results)
        if not results_string:
            results_string = None

        self.results.config(text='Prediction:\n {}'.format(results_string))
        self.results.update_idletasks()
        return

    def uploadFile(self):
        f = filedialog.askopenfilename()
        self.file = f
        _, ext = os.path.splitext(f)
        valid_ext = ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']
        if ext not in valid_ext:
            self.imgPanel.configure(text='Invalid File Type: {}'.format(ext), image='')
            self.imgPanel.image = None

            self.imgPanel.update_idletasks()
            return
        
        else:
            to_predict = ImageTk.PhotoImage(pad_and_resize(f, 400))
            
            self.imgPanel.config(image=to_predict)
            self.imgPanel.image = to_predict
            self.imgPanel.update_idletasks()
            
            self.predict(f)

class classButton(tk.Button):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

class Main_2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        backButton = tk.Button(self, text='Back', command=lambda: controller.show_frame(MainView))
        backButton.pack(side='top', fill='x')
        classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']
        for cls in classes:
            x = copy.deepcopy(cls)
            tk.Button(self, text=cls, command= partial(controller.show_frame,ResultsPage ,x) ).pack(side='bottom', fill='x')

class MainView(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        buttonframe = tk.Frame(self)
        buttonframe.pack(side="top", fill="both", expand=True)
 
        b1 = tk.Button(buttonframe, text="Predict an image", command=lambda: controller.show_frame(Main_1))
        b2 = tk.Button(buttonframe, text="View Validation Results", command=lambda: controller.show_frame(Main_2))

        b1.pack(side="left", fill="both", expand=True)
        b2.pack(side="right",fill="both", expand=True)


class Application(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "PascalVOC Classifier Demo")

        container = tk.Frame(self, width=800, height=600)
        container.pack(side="top", fill='both' , expand = 1)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (MainView, Main_1, Main_2, ResultsPage):

            frame = F(container, self)

            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(MainView)

    def show_frame(self, cont, command=None):
        if command:
            frame = self.frames[cont]
            frame.show(command)
            return
        frame = self.frames[cont]
        frame.lift()


if __name__ == "__main__":
    app = Application()
    app.geometry('800x600')
    app.mainloop()