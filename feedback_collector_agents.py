import csv
import random
import uuid

# Define employer name
employer = "XYZ Corporation"

# Define feedback categories
feedback_categories = ["Management", "Work-Life Balance", "Compensation", "Career Growth", "Company Culture"]

# Define employee names
employees = [f"Employee {i+1}" for i in range(100)]

# Define function to generate random rating
def generate_rating():
    return random.randint(1, 10)

# Define employee agent function
def employee_agent(employee):
    feedback_data = [employee]
    for category in feedback_categories:
        feedback_data.append(str(generate_rating()))
    return feedback_data

# Run employee agents
employee_feedback = []
for employee in employees:
    employee_feedback.append(employee_agent(employee))

# Write to CSV file
with open('employee_feedback.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Employee'] + feedback_categories)
    csvwriter.writerows(employee_feedback)

print("CSV file 'employer_feedback.csv' has been generated.")