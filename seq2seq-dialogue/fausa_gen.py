# -*- coding: UTF-8 -*-
import os
import re

dir_path = os.path.dirname(__file__)
files_dir_path = os.path.join(dir_path, "FAUSA")

files_path = []
for parent, dirnames, filenames in os.walk(files_dir_path):
    for filename in filenames:
        files_path.append(os.path.join(parent, filename))

turns = 0
q_write = file(files_dir_path + "/ques.txt", "wb+")
a_write = file(files_dir_path + "/ans.txt", "wb+")
# q_write = file(files_dir_path + "/slg.ques", "wb+")
# a_write = file(files_dir_path + "/slg.ans", "wb+")
for file_path in files_path:
    file_open = open(file_path, 'r')
    content = file_open.read()
    row = 0
    questions = []
    answers = []
    for sentence in content.split('\n'):
        if not re.compile(r'\s|(Act \d)|(EPISODE|Episode \d)').match(sentence):
            if re.compile(r':').search(sentence):
                row += 1
                if row % 2 == 0:
                    answers.append(sentence)
                else:
                    questions.append(sentence)
        if re.compile(r'Act \d').match(sentence):
            if len(answers)+1 == len(questions):
                questions.pop()
                row -= 1
            questions.append('EOD%s' % turns)
            answers.append('EOD%s' % turns)
            turns += 1
    if len(answers) + 1 == len(questions):
        questions.pop()
        row -= 1
    if len(questions) == len(answers) and len(answers) > 0:
        for i in range(len(questions)):
            if re.compile(r'EOD').match(questions[i]):
                print "here"
                # q_write.write('%s%s' % (questions[i], os.linesep))
                # a_write.write('%s%s' % (answers[i], os.linesep))
            else:
                q_write.write('%s%s' % (questions[i].split(':')[1].strip(), os.linesep))
                a_write.write('%s%s' % (answers[i].split(':')[1].strip(), os.linesep))
    if not len(questions) == len(answers):
        print len(questions)
        print len(answers)
    file_open.close()
q_write.flush()
a_write.flush()
q_write.close()
a_write.close()