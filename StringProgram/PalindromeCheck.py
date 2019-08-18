userinp = raw_input("Enter a String: ")


list1 = []
list1 += userinp;
print list1
# print ord(list1[0]),ord(list1[5])
# print ord(list1[1]),ord(list1[4])
# print ord(list1[2]),ord(list1[3])

b = True;

c =0
for x in range(len(list1)/2):
    if((ord(list1[c]) != ord(list1[len(list1)-c-1]))):
        if (abs(ord(list1[c]) - ord(list1[len(list1) - c - 1])) > 3):
            b = False
            break
        c += 1
    else:
        b = False
        break


print b
# if(len(userinp)%2 == 0):
#     str1 = userinp[0:len(userinp)/2]
#     str2 = userinp[len(userinp):len(userinp)]
#     b = True ;
#     for i,x in enumerate(str1):
#         print i,x
#         #if(x == '('):



