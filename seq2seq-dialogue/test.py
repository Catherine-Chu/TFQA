# -*- coding: UTF-8 -*-
import re

_WORD_END=re.compile(b"([.!?;)(])")
file_open = open('/Users/chuwenjie/PycharmProjects/TFQA/seq2seq-dialogue/FAUSA/ans.txt', 'r')

test=file_open.read()
t=_WORD_END.split(test.strip())
count=0
for i in range(len(t)):
    if not re.match(_WORD_END,t[i]):
        count += 1
        print t[i].strip()
print count
