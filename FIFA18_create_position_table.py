#coding=utf-8

import pymysql
import time

positions = ['gk','lb','cb','rb','lwb','cdm','rwb','lm','cm','rm','cam','lw','cf','rw','st']

#Prepare database
connection = pymysql.connect(
                host='127.0.0.1',
                user='root',
                passwd='123456',
                db='FIFA18',
                charset='utf8'
                )

cursor_1 = connection.cursor() #cursor for complete
cursor_2 = connection.cursor() #cursor for position
time_start = time.time()

#create cmd to fetch data from 'complete' table
cmd = "SELECT ID,"
for pos in positions:
    cmd = cmd+pos+",prefers_"+pos+","
cmd = cmd[:-1]+" FROM complete"
#cmd = cmd+" WHERE overall>=94 "
cmd = cmd+";"

#print cmd
cursor_1.execute(cmd)

#enumerate all players from 'complete' table and insert data to 'position' table accordingly
cnt_player = 0
while True:
    try:
        player = cursor_1.fetchone()
        if player == None:
            break
        cnt_player = cnt_player+1
        #print player
        ID = player[0]
        cmd = "INSERT INTO `position` (`ID`, `pos`, `pos_rate`, `pos_en`) VALUES ("
        cmd = cmd+str(ID)+", "
        cnt_pos = 0
        while cnt_pos*2+1<len(player):
             pos, pos_rate, pos_en = positions[cnt_pos], player[cnt_pos*2+1], player[cnt_pos*2+2]
             cmd_pos = cmd+"'"+pos+"', "+str(pos_rate)+", '"+(pos_en)+"');"
             print(cmd_pos)
             cursor_2.execute(cmd_pos)
             cnt_pos=cnt_pos+1
    except Exception as e:
        print e
        break

time_end = time.time()
print "--PLAYER CNT-- "+str(cnt_player)
print "--TIME SPEND-- "+str(time_end-time_start)

#Leave database
connection.commit()
cursor_1.close()
cursor_2.close()
connection.close()
