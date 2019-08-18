userIn = raw_input("Enter the String: ")

n = int(input("Enter the index to remove: "))

print userIn,n

str = userIn[0:n] + userIn[n+1:len(userIn)]

print (str)

#form