# Student Grade Analyzer


students = {
    "Tim": {
        "Math": 90,
        "Science": 85,
        "English": 88
    },

    "Gim": {
        "Math": 70,
        "Science": 80,
        "English": 75
    },

    "Mim": {
        "Math": 60,
        "Science": 65,
        "English": 70
    }
}


# Function 1: Calculate student averages

def calculate_average(students):

    student_averages = {}

    for student, subjects in students.items():

        total = 0

        for subject, mark in subjects.items():
            total += mark

        average = total / len(subjects)

        student_averages[student] = average

    return student_averages



# Function 2: Determine performance category

def get_category(average):

    if average >= 90:
        return "Excellent"

    elif average >= 75:
        return "Good"

    elif average >= 50:
        return "Average"

    else:
        return "Needs Improvement"



# Function 3: Separate passed and failed students

def get_pass_fail(student_averages):

    passed = {}
    failed = {}

    for student, average in student_averages.items():

        if average >= 50:
            passed[student] = average

        else:
            failed[student] = average

    return passed, failed



# Function 4: Find highest-performing student

def find_highest_student(student_averages):

    highest_student = ""
    highest_average = 0

    for student, average in student_averages.items():

        if average > highest_average:

            highest_average = average
            highest_student = student

    return highest_student, highest_average



# Function 5: Display results

def display_results(student_averages, passed, failed, highest_student, highest_average):

    print("Student Performance Report\n")

    for student, average in student_averages.items():

        category = get_category(average)

        print(f"{student}:")
        print(f"Average: {average:.2f}")
        print(f"Category: {category}\n")


    print("Passed Students:")

    for student, average in passed.items():
        print(f"{student}: {average:.2f}")


    print("\nFailed Students:")

    for student, average in failed.items():
        print(f"{student}: {average:.2f}")


    print(
        f"\nHighest Performer: {highest_student} ({highest_average:.2f})"
    )



# Function 6: Save report to file

def save_report(student_averages, highest_student, highest_average):

    with open("student_report.txt", "w") as file:

        file.write("Student Performance Report\n\n")


        for student, average in student_averages.items():

            category = get_category(average)

            file.write(f"{student}\n")
            file.write(f"Average: {average:.2f}\n")
            file.write(f"Category: {category}\n\n")


        file.write(
            f"Highest Performer: {highest_student} ({highest_average:.2f})"
        )



# Main program

student_averages = calculate_average(students)

passed, failed = get_pass_fail(student_averages)

highest_student, highest_average = find_highest_student(student_averages)

display_results(
    student_averages,
    passed,
    failed,
    highest_student,
    highest_average
)

save_report(
    student_averages,
    highest_student,
    highest_average
)