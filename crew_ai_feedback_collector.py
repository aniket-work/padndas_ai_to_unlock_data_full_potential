from textwrap import dedent
from crewai import Agent, Crew, Task
from langchain_core.tools import Tool
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool
import os
from dotenv import load_dotenv

class FeedbackAgents:
    def __init__(self, use_groq=False):
        load_dotenv()

        # Initialize llm based on the flag
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
        feedback_collection_tool = Tool(
            name="FeedbackCollectionTool",
            description="A tool to collect feedback from employees",
            func=lambda: input("Please provide your feedback:"),
            verbose=True
        )
        return Agent(
            role="Feedback Collector",
            goal=dedent("""
                Gather detailed feedback from employees regarding their experiences,
                satisfaction, and suggestions for improvement.
            """),
            backstory=dedent("""
                You are an expert in conducting interviews and surveys to gather honest
                and comprehensive feedback from employees. Your ability to make employees
                feel comfortable and your keen listening skills ensure you capture valuable
                insights into their experiences and suggestions for improvement.
            """),
            tools=[feedback_collection_tool],
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
            tools=[],
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
            tools=[],
            llm=self.llm,
            verbose=True,
            max_iterations=25
        )

class FeedbackTasks:
    def __init__(self):
        pass

    def feedback_collection_task(self, agent):
        task_description = dedent(f"""\
            You are the Feedback Collector agent responsible for gathering detailed feedback from employees
            regarding their experiences, satisfaction, and suggestions for improvement.
        """)
        return Task(description=task_description, agent=agent, expected_output="")

    def feedback_analysis_task(self, agent):
        task_description = dedent(f"""\
            You are the Feedback Analyst agent responsible for analyzing the collected feedback to identify
            common themes, strengths, and areas for improvement.
        """)
        return Task(description=task_description, agent=agent, expected_output="")

    def feedback_report_task(self, agent):
        task_description = dedent(f"""\
            You are the Feedback Reporter agent responsible for compiling a comprehensive report based on the
            analyzed feedback, highlighting key insights and actionable recommendations.
        """)
        return Task(description=task_description, agent=agent, expected_output="")

# Create agents
agents = FeedbackAgents(use_groq=True)
feedback_collection_agent = agents.feedback_collection_agent()
feedback_analysis_agent = agents.feedback_analysis_agent()
feedback_report_agent = agents.feedback_report_agent()

# Create tasks for each agent
feedback_tasks = FeedbackTasks()
feedback_collection_task = feedback_tasks.feedback_collection_task(feedback_collection_agent)
feedback_analysis_task = feedback_tasks.feedback_analysis_task(feedback_analysis_agent)
feedback_report_task = feedback_tasks.feedback_report_task(feedback_report_agent)

# Create crew
feedback_crew = Crew(
    agents=[feedback_collection_agent, feedback_analysis_agent, feedback_report_agent],
    tasks=[feedback_collection_task, feedback_analysis_task, feedback_report_task],
    verbose=True
)

# Run the feedback process
result = feedback_crew.kickoff()

# Print the results
print("\n\n########################")
print("## Feedback Report")
print("########################\n")
print(result)