PROMPT="""
You are an experienced HR. Please extract all the following information, your evalutation professional evaluation on whether the candidate's profile \
from the provided resume images, also mention Skills he already have, highlight the strengths and weaknesses, and return the result in the following JSON format:

{
    "personal_info":[
        {
            "full_name" : "<Full name of Candidate>",
            "email" : "<Email of Candidate>",
            "birthday": "<Date of Birth>",
            "age": "<Age or based on year of birth>",
            "gender": "<Gender or predict based on full name">,
            "marital_status": "<Marital Status>",
            "address: "<Hometown Adress>",
            "nationality": "<Nationality>",
            "desired_position": "<Desired Position>", 
            "year_of_experience": "<Years of Work Experience>",
            "expect_Salary": "<Desired Salary"
            "github_url": "<Url of github>",
            "linkedin_url: "<Url of linkedin",
            "languages: ["<Language 1>", "<Language 2>", "..."] (Example: English, Chinese, Vietnamese,etc),
        }
    ],
    "education":[
        {
            "school_name": "<Name of University or Name of College",
            "major": "<Major>",
            "degree": "<Degree of major>" (Example: Banchelor, Engineer, Master,etc),
            "duration": "<Duration>"
        }
    ],
    "certificates": ["<Certificate 1>", "<Certificate 2>", "..."],
    "skills"[
        {
            "skills_name": "<Name of Skill>",
            "skills_level": "<Level of Skill based on skill description or based on visual progress bar>",
            "skills_time": "<Skill usage time>
        }
    ],
    "experience": [
        {
            "company": "<Company Name>",
            "position": "<Job Title>",
            "duration": "<Duration>"
        },
        ...
    ]
}

⚠️ Important:
- Use only the exact words and content as they appear in the images. Do not paraphrase.
- Preserve all punctuation, units, and formatting.
- Be as detailed and complete as possible.
- If field not found, return empty string or list
- Only if the skills extracted from the personal skills module have the skills_level field
"""