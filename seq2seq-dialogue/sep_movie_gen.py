# -*- coding=UTF-8 -*-
import os
dlg_data_path=os.path.join(os.path.dirname('__file__'),'movie-dialogs')
data_path=os.path.join(os.path.dirname('__file__'),'tmp')
fw1=file(os.path.join(data_path,'movie_in.txt'),'wb+')
fw2=file(os.path.join(data_path,'movie_out.txt'),'wb+')
fr=open(os.path.join(dlg_data_path,'movie_dialogs.txt'),'r')
line=fr.readline()
input=[]
output=[]
count=0
while line:
    count+=1

    if line !='\n':
        if count%2==0:
            output.append(line)
        else:
            input.append(line)
    else:
        count=0
        if len(input)==len(output)+1:
            newout=input.pop()
            input.append(output[len(output)-1])
            output.append(newout)
    line=fr.readline()
if len(input)==len(output):
    fw1.writelines(input)
    fw2.writelines(output)
else:
    print "wrong"
del input
del output
fw1.flush()
fw2.flush()
fw1.close()
fw2.close()
fr.close()

