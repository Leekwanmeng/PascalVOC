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
# class Page(tk.Frame):
#     def __init__(self, *args, **kwargs):
#         tk.Frame.__init__(self, *args, **kwargs)
#     def show(self):
#         self.lift()

class ResultsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.rv = ResultViewer('D:/Downloads/Deep Learning/Week 6', 'val_set_results3.pt')
        
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
        #self.img.pack(side='top', fill='both', expand=True)
        self.expected = tk.Label(self.sideView, text='Expected:')
        self.actual = tk.Label(self.sideView, text='Actual:')
        self.confidence = tk.Label(self.sideView)
        self.expected.grid(row=1, column=0, sticky='nsew')
        self.actual.grid(row=1, column=1, sticky='nsew')
        
        self.tree = ttk.Treeview(self.panel)
        self.tree.grid(row=0, column=0, sticky='nsw')
        
        scroll = ttk.Scrollbar(self.panel)
        scroll.grid(row=0, column=1, sticky="nsw") # set this to column=2 so it sits in the correct spot.

        scroll.configure(command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)

        #self.sideView = tk.Label(self.panel, text='WTF')

        self.panel.columnconfigure(2, weight=1)
        # self.panel.columnconfigure(1,weight=1)

        self.panel.rowconfigure(0, weight=1)


        #self.tree.pack(side='top', fill='both', expand=True)
        self.tree.bind("<Double-1>", self.onDoubleClick)

    def reset(self):
        print('Deleting children')
        #print(len(self.tree.get_children('')))
        self.tree.delete(*self.tree.get_children(''))
        self.img.config(image='')
        self.img.image = None

    def init_tree(self, cls):
        self.reset()
        #print(self.tree.get_children(''))
        self.cls = cls

        self.img_res = self.rv.get_class_results(cls)
        #print(self.img_res)
        print(cls)
        img_list = self.img_res.index.values

        
        for i in img_list:
            self.tree.insert("",'end', text=i, value=i)

    def onDoubleClick(self, event):
        item = self.tree.selection()[0]
        img_name = self.tree.item(item,"text")
        img_path = self.rv.get_img_path(img_name)

        selected_img = ImageTk.PhotoImage(pad_and_resize(img_path, 400))
        
        self.img.config(image=selected_img)
        self.img.image = selected_img
        self.img.update_idletasks()
            
        #print(img_path)

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
        #openFile.grid(row=0, column=1, sticky='ew')

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
            #to_predict = ImageTk.PhotoImage(resize_and_crop(f, (224,224), crop_type='middle'))
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
        # p1 = Main_1(self)
        # p2 = Main_2(self)
        # rp = ResultsPage(self)

        buttonframe = tk.Frame(self)
        # # container = tk.Frame(self)
        buttonframe.pack(side="top", fill="both", expand=True)
        # container.pack(side="top", fill="both", expand=True)
        # container.grid_rowconfigure(0, weight=1)
        # container.grid_columnconfigure(0, weight=1)

        # p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        # p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        # rp.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        # b1 = tk.Button(buttonframe, text="Predict an image", command=p1.lift)
        # b2 = tk.Button(buttonframe, text="View Validation results", command=p2.lift)
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
            #frame.pack()
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