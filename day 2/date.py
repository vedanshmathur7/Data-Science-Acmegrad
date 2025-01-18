import datetime as dt
now = dt.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d"))

print(now.strftime("%d/%m/%Y, %H:%M:%S"))