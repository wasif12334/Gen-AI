from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader

list_of_urls = [
    # Main Website
    "https://paf-iast.edu.pk/",
    "https://paf-iast.edu.pk/about-paf-iast/",

    # Academics
    "https://paf-iast.edu.pk/bachelor-programs/",
    "https://paf-iast.edu.pk/departments/",

    # Research
    "https://paf-iast.edu.pk/paf-research/",

    # Offices & Resources
    "https://paf-iast.edu.pk/offices-resources/",

    # Admissions
    "https://admissions.paf-iast.edu.pk/",
    "https://admissions.paf-iast.edu.pk/Account/Registration/Signup",

    # Alumni
    "https://alumni.paf-iast.edu.pk/",
    "https://alumni.paf-iast.edu.pk/about.php",
]
data=WebBaseLoader(list_of_urls)

docs=data.load()
print(len(docs))