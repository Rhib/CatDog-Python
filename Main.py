"""Main
    *
    *

    param:
        Author:     Jakob Schmidt & Jonas Bihr
        Date:       06.06.21
        Version:    1.0.0 
        Licencse:   free

    sources:
        none
        
"""

try:
    from Classes.NeuralNetworkVisualizer import *
    from tkinter import *
except ImportError as e:
    logging.error("Class NumberNN Import: The Programm was unable to import all modules\n%s"%(e))



if __name__ == "__main__":
    root = Tk()
    window = neuralNetworkVisualizer(root)
    root.mainloop()