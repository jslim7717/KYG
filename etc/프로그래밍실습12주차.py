# SELF STUDY 10-2
'''
from tkinter import *
from random import *

btnList = [""] * 9
fnameList = ["honeycomb.gif", "icecream.gif", "jellybean.gif", "kitkat.gif", "lollipop.gif", "marshmallow.gif", "nougat.gif", "oreo.gif", "pie.gif"]
photoList=[None] * 9
i, k = 0, 0
xPos, yPos = 0, 0
num = 0

## 메인 코드 부분 ##
window = Tk()
window.geometry("210x210")
shuffle(fnameList)

for i in range(0, 9) :
    photoList[i] = PhotoImage(file = "gif/" + fnameList[i])
    btnList[i] = Button(window, image = photoList[i])  

for i in range(0, 3) :
    for k in range(0, 3) :
        btnList[num].place(x = xPos, y = yPos)
        num += 1
        xPos += 70
    xPos = 0
    yPos += 70

window.mainloop()
'''
# SELF STUDY 10-3
'''
from tkinter import *
from time import *

## 전역  변수 선언 부분 ## 
fnameList = ["jeju1.gif", "jeju2.gif", "jeju3.gif", "jeju4.gif", "jeju5.gif", "jeju6.gif", "jeju7.gif", "jeju8.gif", "jeju9.gif"]
photoList = [None] * 9
num = 0

## 함수 선언 부분 ## 
def clickNext() :
    global num
    num += 1
    if num > 8 :
        num = 0
    photo = PhotoImage(file = "gif/" + fnameList[num])
    pLabel.configure(image = photo)
    pLabel.image = photo
    nameLabel.configure(text=fnamList[num])
    
def clickPrev() :
    global num
    num -= 1
    if num < 0 :
        num = 8
    photo = PhotoImage(file = "gif/" + fnameList[num])
    pLabel.configure(image = photo)
    pLabel.image=photo
    nameLabel.configure(text=fnamList[num])
    nameLabel = label(window, text=frameList[0])
## 메인 코드 부분
window = Tk()
window.geometry("700x500")
window.title("사진 앨범 보기")

btnPrev = Button(window, text = "<< 이전", command = clickPrev)
btnNext = Button(window, text = "다음 >>", command = clickNext)

photo = PhotoImage(file = "gif/" + fnameList[0])
pLabel = Label(window, image = photo)  

btnPrev.place(x = 250, y = 10)
btnNext.place(x = 400, y = 10)
pLabel.place(x = 15, y = 50)

window.mainloop()
'''
# SELF STUDY 10-4
'''
from tkinter import *
from tkinter import messagebox
def keyEvent(event) :
    txt = "눌린키 : Shift + "
    if event.keycode == 37 :
        txt += "왼쪽 화살표"
    elif event.keycode == 38 :
        txt += "위쪽 화살표"
    elif event.keycode == 39 :
        txt += "아래쪽 화살표"
    else :
        txt += "오른쪽 화살표"
    messagebox.showinfo("키보드 이벤트",  txt)
window=Tk()
window.bind("<Key>", keyEvent)
window.bind("<Shift-Up>",keyEvent)
window.mainloop()
'''

# SELF STUDY 11-1
'''
inFp = None
inStr = ""
inFp = open("C:/Temp/data1.txt", "r")
inList = inFp.readlines()
while True :
    inStr = inFp.readline()
    if inStr == " :
        break;
    print(inStr, end="")
inFp.close()
'''
