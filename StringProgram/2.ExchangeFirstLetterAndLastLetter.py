userinp = raw_input("Enter a String:")

result = userinp[-1] + userinp[1:len(userinp)-1]+ userinp[0]

print (result)