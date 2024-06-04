import add_packages
from pprint import pprint
import os
import yaml

from toolkit.langchain import (
  prompts, agents, stores, models, chains, text_embedding_models
)
from toolkit import sql

# *=============================================================================
# call from main
with open(f"{add_packages.APP_PATH}/my_configs/vtc.yaml", 'r') as file:
  configs = yaml.safe_load(file)

llm = models.chat_openai
embeddings = text_embedding_models.OpenAIEmbeddings()
vectorstore = stores.faiss.FAISS


examples_questions_to_sql_onli_course_list = [
    {
        "input": "Which courses are available in the Design category?",
        "query": "SELECT course_name FROM courses WHERE course_category = 'Design';"
    },
    {
        "input": "Who are the instructors for the Personal Development courses?",
        "query": "SELECT DISTINCT instructor_name FROM courses WHERE course_category = 'Personal Development';"
    },
    {
        "input": "Can you provide a summary of the course descriptions for the Digital Marketing category?",
        "query": "SELECT course_name, SUBSTRING(course_description, 1, 100) AS summary FROM courses WHERE course_category = 'Digital Marketing';"
    },
    {
        "input": "Which courses have the longest descriptions?",
        "query": "SELECT course_name, LENGTH(course_description) AS description_length FROM courses ORDER BY description_length DESC LIMIT 5;"
    },
    {
        "input": "How many courses are offered by each instructor?",
        "query": "SELECT instructor_name, COUNT(course_id) AS num_courses FROM courses GROUP BY instructor_name;"
    },
    {
        "input": "Which course categories have the most number of courses?",
        "query": "SELECT course_category, COUNT(course_id) AS num_courses FROM courses GROUP BY course_category ORDER BY num_courses DESC LIMIT 5;"
    },
    {
        "input": "Can you tell me about the course 'Kỹ năng quản lý thời gian'?",
        "query": "SELECT * FROM courses WHERE course_name = 'Kỹ năng quản lý thời gian';"
    },
    {
        "input": "Which courses have the shortest descriptions?",
        "query": "SELECT course_name, LENGTH(course_description) AS description_length FROM courses ORDER BY description_length ASC LIMIT 5;"
    },
    {
        "input": "Which instructors have the most number of courses?",
        "query": "SELECT instructor_name, COUNT(course_id) AS num_courses FROM courses GROUP BY instructor_name ORDER BY num_courses DESC LIMIT 5;"
    },
    {
        "input": "Can you list all the courses that are related to Personal Development?",
        "query": "SELECT course_name FROM courses WHERE course_category = 'Personal Development';"
    }
]

examples_questions_to_sql_vtc_course_list = [
    {
        "input": "What are the different courses offered?",
        "query": "SELECT DISTINCT course_name FROM courses;"
    },
    {
        "input": "Which semester has the most number of courses?",
        "query": "SELECT semester_no, COUNT(course_name) AS num_courses FROM courses GROUP BY semester_no ORDER BY num_courses DESC LIMIT 1;"
    },
    {
        "input": "What is the distribution of course hours across different subjects?",
        "query": "SELECT subject, SUM(hour) AS total_hours FROM courses GROUP BY subject;"
    },
    {
        "input": "Can you provide a breakdown of learning outcomes for each course?",
        "query": "SELECT course_name, learning_outcome FROM courses;"
    },
    {
        "input": "Which courses have the longest learning duration?",
        "query": "SELECT course_name, SUM(hour) AS total_hours FROM courses GROUP BY course_name ORDER BY total_hours DESC;"
    },
    {
        "input": "How many unique learning outcomes are there across all courses?",
        "query": "SELECT COUNT(DISTINCT learning_outcome) AS num_unique_outcomes FROM courses;"
    },
    {
        "input": "Which semester has the highest total number of hours for all courses?",
        "query": "SELECT semester_no, SUM(hour) AS total_hours FROM courses GROUP BY semester_no ORDER BY total_hours DESC LIMIT 1;"
    },
    {
        "input": "Can you tell me about the learning outcomes for the 'Advanced 3D Animation' course?",
        "query": "SELECT learning_outcome FROM courses WHERE course_name = 'Advanced 3D Animation';"
    },
    {
        "input": "Which subject has the most number of courses?",
        "query": "SELECT subject, COUNT(course_name) AS num_courses FROM courses GROUP BY subject ORDER BY num_courses DESC LIMIT 1;"
    },
    {
        "input": "How does the learning outcome 'Ability to research' appear across different courses and subjects?",
        "query": "SELECT course_name, subject FROM courses WHERE learning_outcome LIKE '%Ability to research%';"
    }
]

examples_questions_to_sql = []
examples_questions_to_sql.extend(examples_questions_to_sql_onli_course_list)
examples_questions_to_sql.extend(examples_questions_to_sql_vtc_course_list)

proper_nouns = [
	'Lập trình',
	'Digital Marketing',
	'Trí tuệ nhân tạo A.I',
	'Soft Skills',
	'Personal Development',
	'Design',
	'Ms. Nguyễn Mỹ Hạnh',
	'Ms. Cao Thị Thùy Trang',
	'Ms. Đoàn Phương Trúc',
	'Mr. Dương Hoàng Thanh',
	'Ms. Lương Bảo Trâm',
	'Mr. Nguyễn Phúc Luân',
	'Ms. Lê Thị Minh Thư',
	'Mr. Khang Nguyễn',
	'Ms. My Dương',
	'Ms. Lê Thị Cẩm Thi',
	'Mr. Lê Hoàng Anh Thi',
	'Mr. Đỗ Đức Anh',
	'Mr. Kỷ Thế Vinh\nMr. Lê Vũ Quang',
	'Ms. Nguyễn Hải Uyên',
	'Ms. Trần Thị Thúy Hằng',
	'Mr. Guy-Roger Duvert',
	'Mr. Hoàng Phúc Lộc',
	'Ms. Đinh Hoàng Dung',
	'Ms. Trần Hải Bình',
	'Ms. Nguyễn Tường Vi',
	'Mr. Nguyễn Thiên Ân',
	'Ths. Võ Minh Thành',
	'Ms. Kim Phương',
	'Mr. Lê Hoàng Anh Thi ',
	'Ms. Nguyễn Ngọc Tú Uyên',
	'Mr. Đinh Nguyễn Trọng Nghĩa',
	'Ms. Bùi Vĩnh Nghi',
	'Ms. Bảo Trâm',
	'Ms. Nguyễn Kim Phượng',
	'Ms. Huỳnh Thị Ngọc Tuyền',
	'Mr. Patrick Larochelle',
	'Ms. Lê Diệu Hiền',
	'Ms. Junie Đinh',
	'Ms. Annie Nguyễn',
	'Ms. Tracy Nguyễn',
	'Ms. Trần Nguyễn Hoàng Ngân',
	'Ms. Thanh Trà',
	'Giảng viên Onlinica (AI)',
	'Mr. Nguyễn Đình Cường',
	'Mr. Trịnh Trung Hậu',
	'Mr. Hoàng Khắc Huyên',
	'Ms. Doris',
	'Mr. Thái Bằng Phi (John Thai)',
	'Ms. Đặng Nguyễn Thiên An',
	'Ms. Bùi Thương Huyền',
	'Mr. Huỳnh Nguyên Bảo',
	'Ms. Nguyễn Hải Huyền Trang',
	'Giảng viên Onlinica',
	'Ms. Nguyễn Khánh Ly',
	'Ms. Nguyễn Ngọc Tú Nguyên',
	'Mr. Trần Nguyễn Hoài Nam',
	'Ms. Ngô Thị Ngọc Phượng',
	'Ms. Huỳnh Vũ Thủy Tiên (Tienee)',
	'Mr. Ngô Đình Sơn Thái',
	'Mr. Nguyễn Huỳnh Thiên An',
	'Mr. Bùi Nguyễn Ngọc Dương',
	'Mr. Nguyễn Trọng Tiến',
	'Mr. Trần Anh Tuấn',
	'Ms. Phạm Quỳnh Anh',
	'Ms. Nguyễn Lệ Chi',
	'Ms. Chế Dạ Thảo',
	'ThS. Nguyễn Hải Uyên',
	'Ms. Võ Ngọc Tuyền',
	'Mr. Nguyễn Tấn Huynh',
	'Mr. Nguyễn Đức Giang',
	'Ms. Đỗ Phương Thảo',
	'Ms. Nguyễn Thu Hà',
	'Ms. Trần Thị Phương Thảo',
	'Advanced 3D Animation',
	'Advanced 3D Animation',
	'3D Design',
	'Art Fundamentals',
	'Professional Animation',
	'Advanced Body Mechanics',
	'3D Animation Fundamental',
	'Game Animation',
	'Basic Concept Art',
	'Soft Skills: Creative Thinking, Critical Thinking, Problem Solving, Collaboration, Communication',
	'Semester Orientation Session',
	'Acting Shot',
	'Character Rigging',
	'English IELTS 3',
	'English IELTS 1',
	'English IELTS 2',
	'Dynamic FX',
	'Overview - Semester 2 - 3D Design',
	'Sketchbook',
	'Learning How to Learn',
	'Pantomime',
	'Art theory (Perspective, Color, Proportions & Anatomy of Human)',
	'Personal and Career Development',
	'Drawing',
	'Project 1 (Individual)',
	'Overview - Semester 4 - Professional Animation',
	'Motion Capture',
	'Character Creation',
	'Basic Lighting and Rendering',
	'Storyboard',
	'Capstone Project (Team)',
	'Creature Animation',
	'Compositing Basic',
	'Overview - Semester 3 - Advanced 3D Animation',
	'Texture Painting',
	'Working with Game Engine',
	'Computer Fundamentals',
	'3D Modeling Fundamental',
	'Project 2 (Team)',
	'Grand Project (Team)',
	'Overview - Semester 1 - Art Fundamentals'
 ]

qdrant_txt_vtc_faq = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["vtc_faq"]
)

qdrant_txt_onli_faq = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["onli_faq"]
)

qdrant_lectures_content = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["lectures_content"],
)

my_sql_db = sql.MySQLDatabase()

my_sql_chain = chains.MySqlChain(
	my_sql_db=my_sql_db,
	llm=llm,
	embeddings=embeddings,
	vectorstore=vectorstore,
	proper_nouns=proper_nouns,
	k_retriever_proper_nouns=4,
	examples_questions_to_sql=examples_questions_to_sql,
	k_few_shot_examples=5,
	is_debug=False,
	tool_name="sql_executor",
	tool_description="Generate SQL based on user question and execute it",
	tool_metadata={"data": ["vtc", "onlinica"]},
	tool_tags=["vtc", "onlinica"],
)

tool_chain_sql = my_sql_chain.create_tool_chain_sql()
# *=============================================================================
system_message_vtc = configs["prompts"]["system_message_vtc"]

prompt_onlinica = prompts.create_prompt_tool_calling_agent(
  system_message_vtc
)

tools = [
  # qdrant_lectures_content.retriever_tool,
  qdrant_txt_vtc_faq.retriever_tool,
	qdrant_txt_onli_faq.retriever_tool,
	tool_chain_sql,
]

system_message_custom = configs["prompts"]["system_message_vtc"]
prompt = prompts.create_prompt_tool_calling_agent(system_message_custom)

agent = agents.MyStatelessAgent(
	llm=llm,
	tools=tools,
	prompt=prompt,
	agent_type=configs["agents"]["type"],
	agent_verbose=False,
)