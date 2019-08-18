userinp = raw_input("Enter a string:")

count = 0
for i in userinp:
    if(i == 'a' or i == 'e' or i == 'i' or i == 'o' or i =='u' ):
        count += 1
print (count)

for index,char in enumerate(userinp):
    print (index,char)







