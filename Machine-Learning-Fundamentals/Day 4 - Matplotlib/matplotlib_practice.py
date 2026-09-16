import matplotlib.pyplot as plt

students = ["Tim", "Gim", "Mim", "Kim", "Sim"]

math = [90, 70, 60, 40, 95]
science = [85, 80, 65, 45, 90]
english = [88, 75, 70, 50, 95]

plt.bar(students, math)
plt.title("Student Math Scores")
plt.xlabel("Students")
plt.ylabel("Math Score")
plt.show()

plt.bar(students, science)
plt.title("Student Science Scores")
plt.xlabel("Students")
plt.ylabel("Science Score")
plt.show()

plt.bar(students, english)
plt.title("Student English Scores")
plt.xlabel("Students")
plt.ylabel("English Score")

plt.show()