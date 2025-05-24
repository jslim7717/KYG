##inFp= None
##inStr=''
##
##inFp=open("data1.txt","r")

##inStr=inFp.readline()
##print(inStr,end="")
##
##inStr=inFp.readline()
##print(inStr,end="")
##
##inStr=inFp.readline()
##print(inStr,end="")
##
##inFp.close()

##while True:
##    inStr= inFp.readline()
##    if inStr=="":
##        break;
##    print(inStr,end="")
##    
##inFp.close()

##inList=inFp.readlines()
##for inStr in inList :
##    print(inStr,end="")
##
##inFp.close()

##inFp=None
##fName,inList,inStr="",[],""
##
##fName=input('파일명을 입력하세요:')
##inFp=open(fName,"r")
##
##inList=inFp.readlines()
##for inStr in inList :
##    print(inStr,end="")
##
##inFp.close()

##inFp=None
##inList=""
##
##inFp=open("data1.txt","r")
##
##inList=inFp.readlines()
##for i in range(len(inList)):
##    line=inList[i],rstrip('\n')
##    print(f"{i+1}번째 줄({len(line)}글자):{line}")
##    print(inList)
##print(inList)
##
##inFp.close()

##import os
##
##inFp= None
##fName,inList,inStr="",[],""
##
##fName=input('파일명을 입력하세요:')
##
##if os.path.exists(fName):
##    inFp=open(fName,"r")
##
##    inList=inFp.readlines()
##    for inStr in inList :
##        print(instr,end="")
##
##    inFp.close()
##
##else:
##    print("%s 파일이 없습니다."% fName)

##outFp=None
##outStr=""
##
##outFp=open("data2.txt","w")
##
##while True:
##    outStr=input('내용입력:')
##    if outStr !='' :
##        outFp.writelines(outStr+'\n')
##    else:
##       break
##
##outFp.close()
##print("---정상적으로 파일에 씀---")

##inFp,outFp=None,None
##inStr=""
##
##inFp=open("data1.txt",'r')
##outFp=open('data2.txt','w')
##
##inList=inFp.readlines()
##for inStr in inList :
##    outFp.writelines(inStr)
##
##inFp.close()
##outFp.close()
##print('---파일이 정상적으로 복사되엇음---')

inFp,outFp=None,None
inStr,outStr="",""
i=0
secu=0

secuYN=input("1.암호화 2. 암호 해석 중 선택: ")
inFname=input("입력 파일명을 입력하세요: ")
outFname=input("출력 파일명을 입력하세요: ")

if secuYN=='1':
    secu=100
elif secuYN=='2':
    secu=-100

inFp=open(inFname, 'r', encoding='utf-8')

outFp=open(outFname,'w',encoding='utf-8')

while True:
    inStr=inFp.readline()
    if not inStr:
        break

    outStr=""
    for i in range(0,len(inStr)):
        ch=inStr[i]
        chNum=ord(ch)
        chNum=chNum+secu
        ch2=chr(chNum)
        outStr=outStr+ch2

    outFp.write(outStr)

outFp.close()
inFp.close()
print('%s-->%s 변환 완료' %(inFname, outFname))
