userinp = raw_input("Enter a String: ")

str2 = userinp[::1]
print (str2)
set1 = set()
str1 = ""
for i in range(len(str2)):

    print i
    set1.add(str2[i:i+1])
    if(set1.__contains__(str2[i:i+1]) == False):
        str1 += str2[i:i+1]
    print (str1)



list1 = list(str1)
print (list1)
