from textwrap import dedent
from crewai import Agent, Crew, Task
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool
import os
import csv
import random
from dotenv import load_dotenv

def delegate_work(task: str, context: str, coworker: str) -> str:
    if coworker not in ["Feedback Analyst", "Feedback Reporter"]:
        raise ValueError("Invalid coworker. Must be one of: [Feedback Analyst, Feedback Reporter]")
    return f"Delegating task '{task}' to co-worker '{coworker}' with context: {context}"

delegate_work_tool = Tool(
    name="Delegate work to co-worker",
    description="Delegate a specific task to one of the following co-workers: [Feedback Analyst, Feedback Reporter]",
    func=delegate_work
)

class FeedbackAgents:
    def __init__(self, use_groq=False):
        load_dotenv()
        if use_groq:
            self.llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama3-70b-8192"
            )
        else:
            self.llm = ChatOpenAI(
                model="crewai-llama3-8b",
                base_url="http://localhost:11434/v1",
                api_key="NA"
            )

    def feedback_collection_agent(self):
        return Agent(
            role="Feedback Collector",
            goal=dedent("""
                Gather feedback from employees regarding their experiences,
                satisfaction, and suggestions for improvement.
            """),
            backstory=dedent("""
                You are an expert in conducting interviews and surveys to gather honest
                and comprehensive feedback from employees. Your ability to make employees
                feel comfortable and your keen listening skills ensure you capture valuable
                insights into their experiences and suggestions for improvement.
            """),
            tools=[delegate_work_tool],
            llm=self.llm,
            verbose=True,
            max_iterations=25,
            early_stopping_method="force"
        )

    def feedback_analysis_agent(self):
        return Agent(
            role="Feedback Analyst",
            goal=dedent("""
                Analyze the collected feedback to identify common themes, strengths, and areas for improvement.
            """),
            backstory=dedent("""
                As a seasoned analyst, you possess an unparalleled ability to sift through large volumes
                of feedback to identify patterns and insights. Your analytical skills and attention to detail
                enable you to pinpoint recurring themes and provide a clear summary of the strengths and
                areas needing improvement based on employee feedback.
            """),
            tools=[delegate_work_tool],
            llm=self.llm,
            verbose=True,
            max_iterations=25
        )

    def feedback_report_agent(self):
        return Agent(
            role="Feedback Reporter",
            goal=dedent("""
                Compile a comprehensive report based on the analyzed feedback, highlighting key insights
                and actionable recommendations.
            """),
            backstory=dedent("""
                With your excellent writing skills and ability to present complex information clearly,
                you excel at creating detailed reports that summarize the findings from the feedback analysis.
                Your reports are known for being insightful, well-organized, and actionable, providing valuable
                guidance for organizational improvement.
            """),
            tools=[delegate_work_tool],
            llm=self.llm,
            verbose=True,
            max_iterations=25
        )

class FeedbackTasks:
    def __init__(self):
        pass

    def feedback_collection_task(self, agent):
        task_description = dedent(f"""\
            You are the Feedback Collector agent responsible for gathering feedback from employees
            regarding their experiences, satisfaction, and suggestions for improvement. Your task is to
            generate a CSV file named 'employee_feedback.csv' with the following columns: Employee, Management,
            Work-Life Balance, Compensation, Career Growth, Company Culture. Each row should represent feedback
            from a different employee, with random ratings between 1 and 10 for each column.
        """)
        return Task(description=task_description, agent=agent, expected_output="A CSV file named 'employee_feedback.csv' containing employee feedback data.")

    def feedback_analysis_task(self, agent):
        task_description = dedent(f"""\
            You are the Feedback Analyst agent responsible for analyzing the collected feedback to identify
            common themes, strengths, and areas for improvement. Your input will be the CSV file named
            'employee_feedback.csv' generated by the Feedback Collector agent.
        """)
        return Task(description=task_description, agent=agent, expected_output="A summary of the feedback analysis.")

    def feedback_report_task(self, agent):
        task_description = dedent(f"""\
            You are the Feedback Reporter agent responsible for compiling a comprehensive report based on the
            analyzed feedback, highlighting key insights and actionable recommendations.
        """)
        return Task(description=task_description, agent=agent, expected_output="A comprehensive feedback report.")


agents = FeedbackAgents(use_groq=True)
feedback_collection_agent = agents.feedback_collection_agent()
feedback_analysis_agent = agents.feedback_analysis_agent()
feedback_report_agent = agents.feedback_report_agent()

feedback_tasks = FeedbackTasks()
feedback_collection_task = feedback_tasks.feedback_collection_task(feedback_collection_agent)
feedback_analysis_task = feedback_tasks.feedback_analysis_task(feedback_analysis_agent)
feedback_report_task = feedback_tasks.feedback_report_task(feedback_report_agent)

feedback_crew = Crew(
    agents=[feedback_collection_agent, feedback_analysis_agent, feedback_report_agent],
    tasks=[feedback_collection_task, feedback_analysis_task, feedback_report_task],
    verbose=True
)

result = feedback_crew.kickoff()

print("\n\n########################")
print("## Feedback Report")
print("########################\n")
print(result)