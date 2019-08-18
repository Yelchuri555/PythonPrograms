userinp = raw_input("Enter a String: ")

list1 = userinp.split(' ')

str = ''
count = 0

for i in list1:
    if(len(i)> count):
        count = len(i)
        str = i
print (count,str)