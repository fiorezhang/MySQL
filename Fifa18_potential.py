#coding=utf-8

import pymysql

connection = pymysql.connect(
		host='127.0.0.1',
		user='root',
		passwd='123456',
		db='FIFA18',
		charset='utf8'
		)

cursor = connection.cursor()

potential = input('Query potential above: ')
gap = input('Query gap between potential & overall: ')

sql = 'SELECT name,age,club,overall,potential,eur_value,eur_wage FROM complete WHERE potential>=%s AND potential-overall>%s ORDER BY eur_wage'

cursor.execute(sql, (potential, gap))

while True:
	try:
		player = cursor.fetchone()
       		#print player
		for item in player:
			if type(item) == type(u''):
				item = item.encode('latin1')
				print '%20s' % item,
			else:
				print '%5d' % item,
		print
	except Exception as e:
		print e
		break
		

connection.commit()
cursor.close()
connection.close()
