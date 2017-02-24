# -*- coding: UTF-8 -*-
import os
import re

data_path = os.path.join(os.path.dirname(__file__), "movie-dialogs")
dlg_conv = open(os.path.join(data_path, "movie_conversations.txt"), 'r')
dlg_line = open(os.path.join(data_path, "movie_lines.txt"), 'r')
line = dlg_line.readline()
line_id = {}

while line:
    if len(line.split(' +++$+++ ')) == 5:
        array = line.split(' +++$+++ ')
        key = array[0]
        value = array[4]
        line_id[key] = value
    line=dlg_line.readline()

dlgs = file(os.path.join(data_path, "movie_dialogs.txt"), 'wb+')
content = dlg_conv.readline()
sum_turns=0
count=0
while content:
    if len(content.split(' +++$+++ ')) == 4:
        conv = content.split(' +++$+++ ')[3].strip()
        conv = conv.replace('[', '').replace(']', '').replace('\'', '').split(', ')
        for i in range(len(conv)):
            dlgs.write(line_id[conv[i]])
        dlgs.write(os.linesep)
        sum_turns += len(conv)
        count +=1
    content=dlg_conv.readline()

print float(sum_turns)/count
print count
dlgs.flush()
del line_id
dlgs.close()
dlg_conv.close()
dlg_line.close()

