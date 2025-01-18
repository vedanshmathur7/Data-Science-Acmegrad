# Write a Python program that accepts an integer (no) and computes the value of no+nn+nnn.


n = int (input ("Enter the no. : "))
sum = f"{n*100 + (2*n*10) + (2*n) + n}"

print (sum)

#method 2
# Accept an integer input
n = int(input("Enter an integer (n): "))

# Compute n + nn + nnn
result = n + int(str(n) * 2) + int(str(n) * 3)

# Print the result
print(f"The result of n + nn + nnn for n = {n} is: {result}")

